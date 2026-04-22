#!/bin/bash

###############################################################################
# Usage: pipeline_qgen.sh <source_dir> [sample_size] [transform1 transform2 ...]
#
# source_dir   - Root directory containing numbered case folders
# sample_size  - Number of new cases to copy (default: 10000)
# transform... - Transforms to run (default: transpose pivot unpivot)
#
# Join generation always runs in addition to the listed transforms.
###############################################################################

SOURCE_DIR=${1:?Usage: pipeline_qgen.sh <source_dir> [sample_size] [transform1 transform2 ...]}
SAMPLE_SIZE=${2:-10000}
shift 2
TRANSFORMS=("$@")
[[ ${#TRANSFORMS[@]} -eq 0 ]] && TRANSFORMS=("transpose" "pivot" "unpivot")

ORIGINAL_DIR="./original"

###############################################################################
# Discover which case IDs are already in ORIGINAL_DIR
###############################################################################
mkdir -p "$ORIGINAL_DIR"

declare -A SEEN
while IFS= read -r id; do
  SEEN["$id"]=1
done < <(find "$ORIGINAL_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%P\n')

###############################################################################
# Pick NEW cases, copy only those with relationships.tmdl
###############################################################################
COUNT_NEW=0
declare -a SELECTED_IDS

for id in $(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%P\n' | shuf); do
  [[ ! "$id" =~ ^[0-9]+$ ]] && continue
  [[ -n ${SEEN["$id"]+yes} ]] && continue
  (( COUNT_NEW == SAMPLE_SIZE )) && break

  src="$SOURCE_DIR/$id"
  rel=$(find "$src" -type f -name 'relationships.tmdl' -print -quit)
  [[ -z "$rel" ]] && continue

  dst="$ORIGINAL_DIR/$id"
  mkdir -p "$dst"
  find "$src" -maxdepth 1 -type f -name '*.csv' ! -name '_*' -exec cp {} "$dst/" \;
  cp "$rel" "$dst/"

  SELECTED_IDS+=("$id")
  SEEN["$id"]=1
  ((COUNT_NEW++))
done

echo "Copied $COUNT_NEW new case(s)."

###############################################################################
# Launch qgen jobs
###############################################################################
mkdir -p jobs/qgen
LOG_DIR="logs/qgen"
mkdir -p "$LOG_DIR"

for transform in "${TRANSFORMS[@]}"; do
  JOB="jobs/qgen/nohup_transform_${transform}.sh"
  LOG="$LOG_DIR/nohup_transform_${transform}.log"

  echo "#!/bin/bash" > "$JOB"
  for id in "${SELECTED_IDS[@]}"; do
    echo "python3 training_data_gen_transform.py --id \"$id\" --model o4-mini --transform $transform" >> "$JOB"
  done
  chmod +x "$JOB"
  nohup bash "$JOB" > "$LOG" 2>&1 &
  echo "[LAUNCHED] $JOB -> $LOG"
done

JOIN_JOB="jobs/qgen/nohup_transform_join.sh"
JOIN_LOG="$LOG_DIR/nohup_join.log"
echo "#!/bin/bash" > "$JOIN_JOB"
for id in "${SELECTED_IDS[@]}"; do
  echo "python3 training_data_gen_join.py --id \"$id\" --model o4-mini" >> "$JOIN_JOB"
done
chmod +x "$JOIN_JOB"
nohup bash "$JOIN_JOB" > "$JOIN_LOG" 2>&1 &
echo "[LAUNCHED] $JOIN_JOB -> $JOIN_LOG"
