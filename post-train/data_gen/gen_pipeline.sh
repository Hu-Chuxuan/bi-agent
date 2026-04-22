#!/usr/bin/env bash
set -euo pipefail

# Output directory
OUT_DIR="run_scripts"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

SAMPLE_LIST="selected_case_ids.txt" 

# Configurations: [filename, model, language, script]
configs=(
  "run1_gpt4o_python_tool.sh gpt-4o python run_tool_para.py"
  "run2_gpt4o_sql_tool.sh gpt-4o sql run_tool_para.py"
  "run3_gpt4o_python_notool.sh gpt-4o python run_no_tool_para.py"
  "run4_gpt4o_sql_notool.sh gpt-4o sql run_no_tool_para.py"
  "run5_o4mini_python_notool.sh o4-mini python run_no_tool_para.py"
  "run6_o4mini_sql_notool.sh o4-mini sql run_no_tool_para.py"
  "run5_o4mini_python_tool.sh o4-mini python run_tool_para.py"
  "run6_o4mini_sql_tool.sh o4-mini sql run_tool_para.py"
)

# Initialize all scripts
for cfg in "${configs[@]}"; do
  fname=$(cut -d' ' -f1 <<< "$cfg")
  echo "#!/usr/bin/env bash" > "$OUT_DIR/$fname"
done

# declare -A SAMPLE_OK
while IFS= read -r id; do
  SAMPLE_OK["$id"]=1
done < "$SAMPLE_LIST"

# small helper to grab “first‑two‑dash‑fields” of any string
get_base_id() {
  local str=$1
  echo "${str%%-*}"
}

# Add commands
for raw_id in $(jq -r 'keys[]' training_cases.json); do
  base_id=$(get_base_id "$raw_id")

  [[ -n ${SAMPLE_OK["$base_id"]+yes} ]] || {
      echo "Skipping $raw_id  (ID $base_id not in sample)"
      continue
  }
  for cfg in "${configs[@]}"; do
    read -r fname model lang script <<< "$cfg"
    echo "python3 $script --id \"$raw_id\" --model $model --language $lang --runs 3" >> "$OUT_DIR/$fname"
  done
done

# Make them executable
chmod +x "$OUT_DIR"/*.sh