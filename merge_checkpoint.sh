#!/bin/bash

# Set the checkpoint root directory (contains model_world_size_*.pt and fsdp_config.json)
CKPT_DIR="/HOME/HDD_POOL/ttrl_vision/verl/checkpoints/TTRL-verl/geometry3k-Qwen2.5-VL-7B/0105/TTRL-Len@3k-grpo-112615/global_step_1310"
BASE_MODEL="/HOME/HDD_POOL/Qwen2.5-VL-7B-Instruct"  # Base model path

ln -s "$BASE_MODEL" "$CKPT_DIR/huggingface"

echo "Merging FSDP checkpoints from: $CKPT_DIR"
echo "Target directory: $CKPT_DIR/merged_hf"

# Run the merge script
# Note: this may use a significant amount of CPU memory to load the model
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$CKPT_DIR" \
    --target_dir "$CKPT_DIR/merged_hf"

echo "Merge complete. Please check $CKPT_DIR/merged_hf"
