#!/usr/bin/env bash
# Usage: rft.sh <model_path> <rft_data_dir> [mode]
#   rft_data_dir: path to the rft_data folder (contains py_no_tool/, py_tool/, etc.)
#   mode: py_no_tool | py_tool | sql_no_tool | sql_tool  (default: py_no_tool)

MODEL=${1:?Usage: rft.sh <model_path> <rft_data_dir> [mode]}
RFT_DATA_DIR=${2:?Usage: rft.sh <model_path> <rft_data_dir> [mode]}
MODE=${3:-py_no_tool}

case "$MODE" in
  py_no_tool|py_tool|sql_no_tool|sql_tool)
    DATASET="${RFT_DATA_DIR}/${MODE}/training_case_ids.jsonl" ;;
  *) echo "Error: mode must be one of py_no_tool, py_tool, sql_no_tool, sql_tool" >&2; exit 1 ;;
esac

echo "Starting RFT | model: $MODEL | mode: $MODE"

# 70G*8
RFT_DATA_DIR=$RFT_DATA_DIR \
BI_MODE=$MODE \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    --model_type qwen3 \
    --train_type lora \
    --dataset "$DATASET" \
    --torch_dtype bfloat16 \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_use_double_quant true \
    --bnb_4bit_quant_storage bfloat16 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --save_total_limit 10 \
    --save_steps 5 \
    --logging_steps 5 \
    --output_dir output \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_length 4096 \
    --max_completion_length 4096 \
    --num_generations 2 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --deepspeed zero3 \
    --temperature 0.7 \
    --top_p 0.9 \
    --log_completions true \
    --overlong_filter true \
    --attn_impl flash_attention_2
