# NL2SQL Evaluation

Benchmark runner for evaluating NL-to-SQL / NL-to-Python models on BI tabular queries.

Two evaluation paths are provided:

| Path | Script | Models |
|---|---|---|
| **HuggingFace models** | `run.py` | `infy-32b`, `xiyan-32b`, or any HF model ID |
| **Databao agent** | `databao-agent/bi-bench.py` | Any LLM via the databao LLM config |

---

## HuggingFace Models — `run.py`

Evaluates a local HuggingFace model by loading it on GPU (lazy — loaded on first call).

### Built-in models

| Short name | HuggingFace model ID |
|---|---|
| `infy-32b` | `infly/inf-rl-qwen-coder-32b-2746` |
| `xiyan-32b` | `XGenerationLab/XiYanSQL-QwenCoder-32B-2504` |

You can also pass a full HuggingFace model ID or a local path directly as `--model`.

### Usage

```bash
python run.py \
  --id <case_id> \
  --model <infy-32b|xiyan-32b|HF-model-id|local/path> \
  --language <python|sql> \
  --setting <experiment_tag> \
  --queries-file <path/to/queries.json> \
  --data-dir <path/to/data>
```

| Argument | Description |
|---|---|
| `--id` | Case ID to evaluate |
| `--model` | Short name from the table above, a HuggingFace model ID, or a local path |
| `--language` | `python` or `sql` |
| `--tool` | Flag — use tool-augmented prompts (transform / retrieve / join) |
| `--runs` | Number of attempts per case; stops early on first success (default: 1) |
| `--setting` | Experiment tag used for naming the CSV log and log directory |
| `--queries-file` | Path to a JSON file mapping case IDs to queries |
| `--data-dir` | Directory containing `{id}/` case folders and a `gt/` subfolder |

### Examples

**Run infy-32b on a single case (Python, no tool):**
```bash
python run.py \
  --id 12345 \
  --model infy-32b \
  --language python \
  --setting baseline \
  --queries-file /data/queries.json \
  --data-dir /data/cases
```

**Run xiyan-32b with SQL + tool-augmented prompts:**
```bash
python run.py \
  --id 12345 \
  --model xiyan-32b \
  --language sql \
  --tool \
  --setting tool_eval \
  --queries-file /data/queries.json \
  --data-dir /data/cases
```

### Output

- `{model}-{setting}.csv` — one row per completed case with result and log path
- `logs/{model}-{setting}/{model}_{language}_{tool}_{id}_{timestamp}/` — per-case trace logs

---

## Databao Agent — `databao-agent/bi-bench.py`

Evaluates the databao multi-step agent (transform → retrieve → join → execute) on the BI benchmark.

### Setup

```bash
cd databao-agent
pip install -e .
```

Set the environment variable for token usage logging:
```
TOKEN_CSV_PATH=token_usage.csv   # optional, defaults to token_usage.csv
```

Configure the LLM in `databao/agent/configs/llm.py` by setting:
```
AZURE_OPENAI_GPT_ENDPOINTS='["https://your-endpoint.openai.azure.com/"]'
```
Authentication uses Azure CLI credentials (`az login`).

### Usage

```bash
python bi-bench.py \
  --queries-file <path/to/queries.json> \
  --data-dir <path/to/data> \
  --model <model_name> \
  --runs <N> \
  --result-csv results.csv \
  --log-dir logs
```

| Argument | Description |
|---|---|
| `--queries-file` | JSON file mapping case IDs to queries |
| `--data-dir` | Directory containing case folders and a `gt/` subfolder |
| `--model` | LLM name passed to databao LLMConfig (e.g. `gpt-4o`) |
| `--temperature` | Sampling temperature (default: 0) |
| `--runs` | Runs per case for pass@k estimation (default: 10) |
| `--result-csv` | Output CSV path |
| `--log-dir` | Directory for per-case `.log` files |

---

## Data Format

**`queries.json`** — maps case IDs to query strings:
```json
{
  "12345": { "query": "What is the total revenue by region?" },
  "67890": { "query": "List the top 5 products by sales." }
}
```

**Case folder layout:**
```
<data-dir>/
  12345/
    table1.csv
    table2.csv
  67890/
    ...
  gt/
    12345.csv     # or 12345_0.csv, 12345_1.csv, ... for multiple acceptable answers
    67890.csv
```
