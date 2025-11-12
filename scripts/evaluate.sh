#!/bin/bash
export PYTHONPATH='.'

base_model=$1 # e.g., decapoda-research/llama-7b-hf
tune_ckpt_name=$2 
prune_ckpt=$3
epochs=("${@:4}")

for epoch in "${epochs[@]}"; 
do
    cp $tune_ckpt_name/adapter_config.json $tune_ckpt_name/checkpoint-$epoch/
    mv $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin $tune_ckpt_name/checkpoint-$epoch/adapter_model.bin

    tune_id="${tune_ckpt_name##*/}"
    python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name/checkpoint-$epoch,config_pretrained=$base_model --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --output_path results/${tune_id}_$epoch.json --no_cache
done

#python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=prune_log/llama_prune_0.5/pytorch_model.bin,peft=tune_log/llama_0.5,config_pretrained=/the/model/path/ --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --output_path results/tune_0.5.json --no_cache