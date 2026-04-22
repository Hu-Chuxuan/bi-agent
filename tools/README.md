# BI-Agent Evaluation

This directory contains the main BI-Agent evaluation framework used to reproduce the results in the paper. It runs LLMs on the BIBench benchmark with or without the data-management tool suite.

## Directory Layout

```
tools/
├── run_large_models.py        # runner for proprietary / open-source LLMs
├── run_posttrained.py         # runner for locally-loaded post-trained models
├── data_management_tools/     # ground-truth tool implementations
│   ├── join/                  # pre-computed join relationships per case
│   ├── transformation/        # pre-computed table transformations per case
│   └── metadata/              # pre-computed table metadata per case
└── utils/
    ├── llm_client.py          # unified LLM client (configure your endpoint here)
    ├── llm_posttrained.py     # local HF model loader (reads POSTTRAINED_MODEL_PATH)
    ├── caller_large_models.py # thin wrapper that routes to llm_client
    ├── caller_posttrained.py  # thin wrapper that routes to llm_posttrained
    ├── prompt.py              # all system / user prompt templates
    ├── py_tools.py            # Python code-execution + data-management tools
    ├── sql_tools.py           # SQL code-execution + data-management tools
    └── file2var.py            # table-name sanitization utilities
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt   # or: pip install openai azure-ai-inference pandas numpy python-dotenv
```

### 2. Configure the LLM client

Open `utils/llm_client.py` and set the `client` variable to your Azure OpenAI / AI Foundry instance:

```python
# utils/llm_client.py
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version="2025-01-01-preview",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-key",           # or use azure_ad_token_provider for Entra ID auth
)
```

Supported models in `run_large_models.py`:

| Model name | API type |
|---|---|
| `gpt-4o` | Chat Completions |
| `o4-mini` | Chat Completions |
| `gpt-5.2` | Responses API |
| `Llama-4-Maverick-17B-128E-Instruct-FP8` | Azure AI Inference |

### 3. Configure the benchmark path

Both runners resolve the benchmark directory relative to this folder. The expected layout (provided by `../bi-bench/`) is:

```
../bi-bench/
├── {case_id}/        # CSV tables for this case
├── gt/               # ground-truth CSVs (one or more per case)
└── queries.json      # maps case IDs to natural-language queries
```

---

## Running Evaluations

Both scripts share the same interface. The only difference is that `run_large_models.py` takes `--model` explicitly, while `run_posttrained.py` reads the model path from the environment.

### Proprietary / open-source models

```bash
python run_large_models.py \
  --id <case_id> \
  --model <gpt-4o|o4-mini|gpt-5.2|Llama-4-Maverick-17B-128E-Instruct-FP8> \
  --language <python|sql> \
  [--tool] \
  [--runs N] \
  [--log-csv results.csv] \
  [--log-dir logs]
```

### Post-trained models

```bash
POSTTRAINED_MODEL_PATH=/path/to/checkpoint \
python run_posttrained.py \
  --id <case_id> \
  --language <python|sql> \
  [--tool] \
  [--runs N] \
  [--log-csv results_posttrained.csv] \
  [--log-dir logs]
```

### Arguments

| Argument | Description |
|---|---|
| `--id` | Case ID (folder name under `../bi-bench/`) |
| `--model` | Model name (large models only) |
| `--language` | `python` or `sql` |
| `--tool` | Enable the full data-management tool suite (transform / retrieve / join) |
| `--runs` | Attempts per case; stops early on first success (default: 1) |
| `--log-csv` | CSV file to append one summary row per case |
| `--log-dir` | Directory for full per-case trace logs |

### Examples

```bash
# GPT-4o, Python, no tools (baseline)
python run_large_models.py --id 101248502 --model gpt-4o --language python

# o4-mini, SQL, with tool suite
python run_large_models.py --id 101248502 --model o4-mini --language sql --tool

# Post-trained model, Python, with tool suite, 3 attempts
POSTTRAINED_MODEL_PATH=/checkpoints/rft/py_tool \
python run_posttrained.py --id 101248502 --language python --tool --runs 3
```

---

## Evaluation Modes

| Mode | `--language` | `--tool` | Description |
|---|---|---|---|
| Python baseline | `python` | off | Free-form Python code execution only |
| Python + tools | `python` | on | Python + transform / retrieve / join |
| SQL baseline | `sql` | off | Free-form SQLite execution only |
| SQL + tools | `sql` | on | SQL + transform / retrieve / join |

## Data-Management Tools

When `--tool` is set, three additional tools are available to the model before it writes code:

| Tool | Implementation | Description |
|---|---|---|
| `transform_tables` | Pre-computed | Applies transpose / pivot / unpivot transformations; outputs live in `data_management_tools/transformation/` |
| `retrieve_relevant_tables` | Real-time GPT-4o call | Queries GPT-4o to identify which tables are relevant to the user query |
| `discover_join_relationships` | Pre-computed | Returns joinable column pairs and merge statements; outputs live in `data_management_tools/join/` |

