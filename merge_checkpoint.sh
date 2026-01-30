#!/bin/bash

# 设置 Checkpoint 根目录 (包含 model_world_size_*.pt 和 fsdp_config.json 的目录)
CKPT_DIR="/HOME/HDD_POOL/ttrl_vision/verl/checkpoints/TTRL-verl/geometry3k-Qwen2.5-VL-7B/0105/TTRL-Len@3k-grpo-112615/global_step_1310"
BASE_MODEL="/HOME/HDD_POOL/Qwen2.5-VL-7B-Instruct"  # 基座模型路径

ln -s "$BASE_MODEL" "$CKPT_DIR/huggingface"

echo "Merging FSDP checkpoints from: $CKPT_DIR"
echo "Target directory: $CKPT_DIR/merged_hf"

# 运行合并脚本
# 注意：这需要消耗一定的 CPU 内存来加载模型
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$CKPT_DIR" \
    --target_dir "$CKPT_DIR/merged_hf"

echo "Merge complete. Please check $CKPT_DIR/merged_hf"
