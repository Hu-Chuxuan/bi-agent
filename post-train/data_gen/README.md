# Data Generation Pipeline

This directory generates the training data (SFT and RFT) for AutoBI.

## Source Data

The pipeline expects a directory of BI case folders. Each folder is a numeric ID containing:
- One or more `.csv` table files
- A `relationships.tmdl` file describing table relationships

This source data is not included in the repo. Pass its path as the first argument to `pipeline_qgen.sh`.

## Pipeline Stages

### 1. Q-Gen — Sample cases and generate questions

```bash
./pipeline_qgen.sh <source_dir> [sample_size] [transform1 transform2 ...]
```

- Copies `sample_size` new cases from `<source_dir>` into `./original/`
- Runs `training_data_gen_transform.py` (o4-mini) for each transform (`transpose`, `pivot`, `unpivot`)
- Runs `training_data_gen_join.py` (o4-mini) for join transforms
- Outputs `training_cases_{transform}.json` — maps case IDs to queries

Example:
```bash
./pipeline_qgen.sh /data/bi_cases 10000
./pipeline_qgen.sh /data/bi_cases 1000000 unpivot  # large unpivot-only run
```

### 2. Trace — Run models to generate solution traces

```bash
./pipeline_trace.sh
```

Runs `run.py` with `gpt-4o` for each combination of language × tool-mode × transform.
Successful traces are saved to `raw_traces/{model}_{language}_{tool|no_tool}/{transform}/{id}.json`.

Run a single trace manually:
```bash
python3 run.py --id 12345 --model gpt-4o --language python --transform transpose --runs 3
python3 run.py --id 12345 --model gpt-4o --language sql --transform pivot --tool --runs 3
```

Args:
- `--tool` — use tool-augmented prompts (default: baseline prompts)
- `--data-dir` — directory containing `cases_*` and `training_cases_*.json` (default: `.`)
- `--output-dir` — root directory for saving traces (default: `raw_traces`)

### 3. Transform Traces — Convert to training format

```bash
python3 transform_trace.py --filename raw_traces/gpt-4o_python_no_tool/transpose/12345.json
```

Outputs to `transformed_traces/{...same structure...}`.

Override folder names with `--raw-dir` and `--output-dir`.

### 4. Gen JSONL — Build per-turn SFT samples

```bash
python3 gen_jsonl.py --filename transformed_traces/gpt-4o_python_no_tool/transpose/12345.json
```

Outputs one `.jsonl` file per trace under `sft_data/`.

Override with `--transformed-dir` and `--output-dir`.

Alternatively, use `gen_jsonl_overall.py` to produce a single JSONL per full trace instead of per-turn.

### 5. Combine — Merge all JSONL files

```bash
python3 combine_cases.py
```

Produces the final combined JSONL files for SFT training.

## Environment Variables

Set these in a `.env` file before running:

| Variable | Used by | Description |
|---|---|---|
| `AZURE_OPENAI_O_CONFIGS` | `utils/llm_client.py` | JSON array of `{"endpoint":"...","key":"..."}` for o4-mini (query generation) |
| `AZURE_OPENAI_GPT_CONFIGS` | `utils/llm_client.py` | JSON array of `{"endpoint":"...","key":"..."}` for gpt-4o (trace generation) |

Example `.env`:
```
AZURE_OPENAI_O_CONFIGS='[{"endpoint":"https://my-resource.openai.azure.com/","key":"abc123"}]'
AZURE_OPENAI_GPT_CONFIGS='[{"endpoint":"https://my-resource.openai.azure.com/","key":"abc123"}]'
```

## Directory Structure

```
data_gen/
├── original/                    # copied BI case folders (created by pipeline_qgen.sh)
├── cases_{transform}/           # generated cases per transform
│   ├── {id}/                    # CSV files for this case
│   └── gt/                      # ground-truth CSVs
├── training_cases_{transform}.json  # query map for each transform
├── raw_traces/                  # successful model traces
│   └── {model}_{lang}_{mode}/{transform}/{id}.json
├── transformed_traces/          # traces converted to training format
├── sft_data/                    # per-turn SFT JSONL files
├── utils/                       # shared utilities
├── run.py                       # trace runner
├── transform_trace.py           # trace format converter
├── gen_jsonl.py                 # per-turn JSONL builder
├── gen_jsonl_overall.py         # full-trace JSONL builder
├── combine_cases.py             # final JSONL combiner
├── pipeline_qgen.sh             # stage 1: sample and generate questions
└── pipeline_trace.sh            # stage 2: run model traces
```
