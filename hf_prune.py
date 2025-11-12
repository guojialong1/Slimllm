import gc
import random
import argparse
from typing import Tuple, Optional

import torch
import numpy as np
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig, AutoTokenizer
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP

from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts

from adapters.model_adapter import LayerAdapter, ModelAdapter
from adapters.llama_adapter import LlamaLayerAdapter, LlamaModelAdapter
import inspect
import logging


class PruneLayer(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (None,)

        if use_cache:
            outputs += (None,)

        return outputs


def cleanup_memory() -> None:
    """Run GC and clear GPU memory."""
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        logging.debug(
            f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
            f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
        )


def map_tensors(obj, device: torch.device = None, dtype: torch.dtype = None):
    """Recursively map tensors to device and dtype."""
    if isinstance(obj, torch.Tensor):
        if device is not None:
            obj = obj.to(device=device)
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(map_tensors(x, device, dtype) for x in obj)
    elif isinstance(obj, dict):
        return {k: map_tensors(v, device, dtype) for k, v in obj.items()}  # type: ignore
    else:
        return obj


def get_layer0_inputs(model_adapter: ModelAdapter, batch):
    """
    Returns the inputs to the first layer of the model (after embeddings).

    Also returns the additional args and kwargs that are passed to
    the first layer (such as the attention mask, or caches K/V values).

    This relies on all arguments to subsequent layers being the same.

    NB: this won't work from OPT 350m.
    """
    # Move embeddings to device.
    # for W in model_adapter.get_embeddings():
    #     W.weight = torch.nn.Parameter(W.weight)

    class Catcher(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, *args, **kwargs):
            self.saved_args = args
            self.saved_kwargs = kwargs
            raise ValueError

    layer0_adapter = model_adapter.get_layers()[0]
    layer0_catcher = Catcher()
    model_adapter.set_raw_layer_at(0, layer0_catcher)

    try:
        model_adapter.model(batch, labels=batch)
    except ValueError:
        pass

    # grab the inputs and caught arguments
    args = layer0_catcher.saved_args
    kwargs = layer0_catcher.saved_kwargs

    # put the caught stuff on cpu
    # args = map_tensors(args, device='cpu')
    # kwargs = map_tensors(kwargs, device='cpu')

    # put the layer back to normal
    model_adapter.set_raw_layer_at(0, layer0_adapter.layer)

    # Move embeddings back to cpu, and clear GPU cache.
    # for W in model_adapter.get_embeddings():
    #     W.weight = torch.nn.Parameter(W.weight.to('cpu'))

    # Run GC and cleanup GPU memory
    cleanup_memory()

    return args, kwargs


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name), 
        config=args.__dict__,
                root_dir='prune_log',
        setup_sublogger=True
    )

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        low_cpu_mem_usage=True if args.torch_version >=1.9 else False
    )
    if args.device != "cpu":
        model.half()
    model.to(args.device)

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.slimllm:

        example_prompts = get_examples('bookcorpus', tokenizer, args.num_examples, seq_len=128).to(args.device)
        model_adapter = LlamaModelAdapter(model)
        model_adapter.model.zero_grad()
        layers = model_adapter.get_layers()
        args2, kwargs = [], []
        with torch.no_grad():
            batch_input = example_prompts
            args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch_input)
            args2.append(args_batch)
            kwargs.append(kwargs_batch)

            layer_score = []
            args_copy = args2[:]
            for i, layer in enumerate(layers):
                layer = layer.layer
                layer_out_list = []
                print(i)
                for args_batch, kwargs_batch in zip(args_copy, kwargs):
                    layer_out = layer(*args_batch, **kwargs_batch)
                    layer_out_list.append((layer_out[0],))
                    score = torch.cosine_similarity(layer_out[0], args_batch[0], dim=-1).mean()
                    print(score)
                layer_score.append(score)
                args_copy = layer_out_list

            layer_score = torch.tensor(layer_score)
            value, index = torch.sort(layer_score, -1, descending=True)

            index = index[:args.pruned_layers]
            value = value[:args.pruned_layers]

            value = (value * args.alpha).softmax(-1).tolist()
            d = {}
            for i, x in enumerate(index.tolist()):
                d[x] = value[i] * len(index)

            for i, layer in enumerate(layers):
                layer = layer.layer
                layer_out_list = []

                if i in index:
                    print(i)
                    for args_batch, kwargs_batch in zip(args2, kwargs):
                        kwargs_batch['prune_ratio'] = args.pruning_ratio * d[i]
                        layer_out = layer.prune(*args_batch, **kwargs_batch)
                        del kwargs_batch['prune_ratio']
                        layer_out_list.append((layer_out[0],))
                else:
                    for args_batch, kwargs_batch in zip(args2, kwargs):
                        layer_out = layer(*args_batch, **kwargs_batch)
                        layer_out_list.append((layer_out[0],))
                args2 = layer_out_list

        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    else:
        raise NotImplementedError
    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
    
    gc.collect()
    torch.cuda.empty_cache()

    if args.save_model:
        model.half()
        torch.save({
            'model': model, 
            'tokenizer': tokenizer,
        }, logger.best_checkpoint_path)
    
    if args.eval_device != "cpu":
        model.half()
    model.to(args.eval_device)

    model.config.pad_token_id = tokenizer.pad_token_id = 0 
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if args.test_after_train:
        logger.log("\n==================Generation Results After Pruning================\n")
        
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.eval_device)

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                
                result = tokenizer.decode(generation_output[0])
                logger.log(result)
        
        logger.log("\n==================Finish================\n")
    
    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.eval_device)
    logger.log("PPL after pruning: {}".format(ppl))
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="/path/of/model/", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')

    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

    parser.add_argument('--slimllm', action='store_true', help='use SlimLLM')
    parser.add_argument('--pruned_layers', type=int, help='the number of layers for pruning', default=30)
    parser.add_argument('--num_examples', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=10)

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')

    parser.add_argument('--seed', type=int, default=42, help='seed')  # 42
    parser.add_argument('--save_model', action='store_true', help='if save model')
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
