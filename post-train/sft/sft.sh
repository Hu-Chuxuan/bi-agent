#!/usr/bin/env bash
# Usage: sft.sh <model> <sft_data_dir> <output_dir> [mode]
#   model:       HF model name or local path
#   sft_data_dir: path to the SFT data folder (contains py_no_tool/, py_tool/, etc.)
#   output_dir:  base path for saving checkpoints
#   mode:        py_no_tool | py_tool | sql_no_tool | sql_tool | all  (default: all)

MODEL=${1:?Usage: sft.sh <model> <sft_data_dir> <output_dir> [mode]}
SFT_DATA_DIR=${2:?Usage: sft.sh <model> <sft_data_dir> <output_dir> [mode]}
OUTPUT_DIR=${3:?Usage: sft.sh <model> <sft_data_dir> <output_dir> [mode]}
MODE=${4:-all}

ALL_MODES=(py_no_tool py_tool sql_no_tool sql_tool)

case "$MODE" in
  py_no_tool|py_tool|sql_no_tool|sql_tool) MODES=("$MODE") ;;
  all) MODES=("${ALL_MODES[@]}") ;;
  *) echo "Error: mode must be one of py_no_tool, py_tool, sql_no_tool, sql_tool, all" >&2; exit 1 ;;
esac

for mode in "${MODES[@]}"; do
  echo "Training variant: $mode"

  CUDA_VISIBLE_DEVICES=0 swift sft \
    --model "$MODEL" \
    --train_type lora \
    --dataset "${SFT_DATA_DIR}/${mode}" \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --save_steps 100 \
    --save_total_limit 20 \
    --logging_steps 5 \
    --max_length 16384 \
    --output_dir "${OUTPUT_DIR}/${mode}" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_name "$mode"
done
