prune_ckpt_path='vicuna_prune_7b_0.2'
tune_ckpt_path='vicuna_7b_0.2'

echo "[START] - Start Pruning Model"
python hf_prune.py --base_model /vicuna/ckpt/path --pruning_ratio 0.25 --device cpu  --eval_device cuda --slimllm --pruned_layers 26 --alpha 10 --save_ckpt_log_name $prune_ckpt_path --test_after_train --save_model
echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"
CUDA_VISIBLE_DEVICES=0 python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune --lora_r 4 --num_epochs 2 --learning_rate 1e-4 --batch_size 16
echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

prune_ckpt_path='vicuna_prune_7b_0.5'
tune_ckpt_path='vicuna_7b_0.5'

echo "[START] - Start Pruning Model"
python hf_prune.py --base_model /vicuna/ckpt/path --pruning_ratio 0.55 --device cpu  --eval_device cuda --slimllm --pruned_layers 30 --alpha 7 --save_ckpt_log_name $prune_ckpt_path --test_after_train --save_model
echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"
CUDA_VISIBLE_DEVICES=0 python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune --lora_r 4 --num_epochs 2 --learning_rate 1e-4 --batch_size 16
echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"