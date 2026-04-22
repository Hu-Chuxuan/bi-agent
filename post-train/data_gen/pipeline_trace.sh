#!/bin/bash

TRANSFORMS=("transpose" "pivot" "join")
MODELS=("o4-mini" "gpt-4o")
LANGUAGES=("python" "sql")
TOOL_MODES=("tool" "no_tool")
LOG_DIR="./logs/trace"
mkdir -p "$LOG_DIR"
JOB_DIR="./jobs/trace"
mkdir -p "$JOB_DIR"

RECORD_FILE="run_record.txt"

for transform in "${TRANSFORMS[@]}"; do
  INPUT_FILE="training_cases_${transform}.json"
  if [[ ! -f "$INPUT_FILE" ]]; then
    echo "[WARNING] File $INPUT_FILE not found, skipping..."
    continue
  fi

  # GPT-4o: one job per transform (all languages/modes)
  JOB_SH="$JOB_DIR/nohup_run_gpt4o_${transform}.sh"
  LOG_FILE="$LOG_DIR/nohup_run_gpt4o_${transform}.log"
  echo "#!/bin/bash" > "$JOB_SH"
  for raw_id in $(jq -r 'keys[]' "$INPUT_FILE"); do
    for lang in "${LANGUAGES[@]}"; do
      for mode in "${TOOL_MODES[@]}"; do
        ENTRY="gpt-4o,$lang,$mode,$transform,$raw_id"
        if grep -Fxq "$ENTRY" "$RECORD_FILE" 2>/dev/null; then
          echo "[SKIPPED] $ENTRY"
          continue
        fi
        TOOL_FLAG=""
        [[ "$mode" == "tool" ]] && TOOL_FLAG="--tool"
        echo "python3 run.py $TOOL_FLAG --id \"$raw_id\" --model gpt-4o --language $lang --runs 3 --transform $transform && echo \"$ENTRY\" >> $RECORD_FILE" >> "$JOB_SH"
      done
    done
  done
  chmod +x "$JOB_SH"
  nohup bash "$JOB_SH" > "$LOG_FILE" 2>&1 &
  echo "[LAUNCHED] $JOB_SH -> $LOG_FILE"

  # o4-mini: one job per (lang, mode) for parallelism
  for lang in "${LANGUAGES[@]}"; do
    for mode in "${TOOL_MODES[@]}"; do
      JOB_SH="$JOB_DIR/nohup_run_o4mini_${transform}_${lang}_${mode}.sh"
      LOG_FILE="$LOG_DIR/nohup_run_o4mini_${transform}_${lang}_${mode}.log"
      echo "#!/bin/bash" > "$JOB_SH"
      for raw_id in $(jq -r 'keys[]' "$INPUT_FILE"); do
        ENTRY="o4-mini,$lang,$mode,$transform,$raw_id"
        if grep -Fxq "$ENTRY" "$RECORD_FILE" 2>/dev/null; then
          echo "[SKIPPED] $ENTRY"
          continue
        fi
        TOOL_FLAG=""
        [[ "$mode" == "tool" ]] && TOOL_FLAG="--tool"
        echo "python3 run.py $TOOL_FLAG --id \"$raw_id\" --model o4-mini --language $lang --runs 3 --transform $transform && echo \"$ENTRY\" >> $RECORD_FILE" >> "$JOB_SH"
      done
      chmod +x "$JOB_SH"
      nohup bash "$JOB_SH" > "$LOG_FILE" 2>&1 &
      echo "[LAUNCHED] $JOB_SH -> $LOG_FILE"
    done
  done
done
