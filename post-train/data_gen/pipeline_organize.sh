#!/bin/bash

###############################################################################
# 2) transform_trace.py  ── raw traces
###############################################################################
#!/bin/bash

# INPUT_DIR="./raw_traces_1213"

# find "$INPUT_DIR" -type f -name "*.json" | while read -r file; do
#     echo "Transforming $file"
#     python3 transform_trace.py --filename "$file"
# done

###############################################################################
# 3) gen_jsonl.py  ── transformed traces
###############################################################################
INPUT_DIR="./transformed_traces_1213"
find "$INPUT_DIR" -type f -name "*.json" | while read -r file; do
    echo "Dumping $file to jsonl"
    python3 gen_jsonl.py --filename "$file"
done
