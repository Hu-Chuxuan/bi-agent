# NL2SQL Evaluation

Benchmark runner for evaluating NL-to-SQL / NL-to-Python models on BI tabular queries.

Models are evaluated via `run.py`:

| Path | Script | Models |
|---|---|---|
| **HuggingFace models** | `run.py` | `infy-32b`, `xiyan-32b`, `kwai-autosql-14b`, `kwai-autosql-32b`, or any HF model ID |
| **External agents** | `agent_harness.py` | ktx, Databao Agent (install upstream, then implement the adapter) |

---

## HuggingFace Models — `run.py`

Evaluates a local HuggingFace model by loading it on GPU (lazy — loaded on first call).

### Built-in models

| Short name | HuggingFace model ID |
|---|---|
| `infy-32b` | `infly/inf-rl-qwen-coder-32b-2746` |
| `xiyan-32b` | `XGenerationLab/XiYanSQL-QwenCoder-32B-2504` |
| `kwai-autosql-14b` | `Kwai-AutoSQL/Kwai-AutoSQL-14B` |
| `kwai-autosql-32b` | `Kwai-AutoSQL/Kwai-AutoSQL-32B` |

You can also pass a full HuggingFace model ID or a local path directly as `--model`.

### Usage

```bash
python run.py \
  --id <case_id> \
  --model <infy-32b|xiyan-32b|kwai-autosql-14b|kwai-autosql-32b|HF-model-id|local/path> \
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

## Agent baselines: — Databao Agent & ktx

The **Databao Agent** and **ktx** baselines are external systems and are not vendored in this repository. The adapters in [`agent_harness.py`](agent_harness.py) (`KtxAgent`, `DatabaoAgent`) are already implemented — you only need to install the underlying agent and point the harness at it. The harness builds the SQLite database, invokes the agent, grades against the ground truth, and logs a per-case success rate.

**ktx.** Install ktx per <https://github.com/Kaelio/ktx> (Node 22, `npm install -g @kaelio/ktx`, `ktx admin runtime install --feature local-embeddings`) so the `ktx` CLI is on your `PATH`, then `pip install openai` for the answering model:

```bash
export OPENAI_API_KEY=...      
python agent_harness.py --agent ktx --id 101248502 \
  --queries-file ../bi-bench/queries.json --data-dir ../bi-bench \
  --log-csv ../results/ktx.csv
```

ktx's own semantic layer runs on gpt-5.5 via the `codex` backend (set in `KtxAgent`); the query is answered by `--model`.

**Databao Agent.** Install it from its upstream repository (it ships its own `bi-bench.py` runner), then point the adapter at that script:

```bash
export DATABAO_BIBENCH=/path/to/databao/bi-bench.py
python agent_harness.py --agent databao --id 101248502 --model gpt-5.2 \
  --queries-file ../bi-bench/queries.json --data-dir ../bi-bench \
  --log-csv ../results/databao.csv
```

Because databao ships a complete benchmark runner, you can also run its `bi-bench.py` directly; adjust `DatabaoAgent.run` if your databao version's flags/output differ.

Their pre-computed BI-Bench scores are in [`../results/nl2sql_systems.csv`](../results/nl2sql_systems.csv) (the `databao` and `ktx` columns).

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
