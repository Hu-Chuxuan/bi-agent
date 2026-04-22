# BI-Agent

This repository contains the reproduction artifact for the SIGMOD 2027 Round 2 submission (Paper ID 406): **BI-Agent: Automating End-to-end Business Intelligence**


## Repository Structure

```
.
├── bi-bench/       # BIBench dataset
├── tools/          # BI-Agent evaluation framework
├── nl2sql/         # NL2SQL system evaluation (HF models + databao agent)
└── post-train/     # Post-training pipeline (data generation, SFT, RFT)
```

---

## Components

### `bi-bench/` — The BIBench Benchmark

The benchmark dataset. Each case is a real-world BI project consisting of multiple CSV tables, a relationship file, and a natural-language business question extracted from a user dashboard.

```
bi-bench/
├── {case_id}/      # CSV tables for this case
├── gt/             # ground-truth answer CSVs
└── queries.json    # maps case IDs to queries
```

---

### `tools/` — BI-Agent Evaluation

The main evaluation framework. Runs any LLM on BIBench cases with or without the data-management tool suite (transform, retrieve, join).

Two runners:

| Script | Models |
|---|---|
| `run_large_models.py` | Proprietary / large open-source: `gpt-4o`, `o4-mini`, `gpt-5.2`, `Llama-4-Maverick` |
| `run_posttrained.py` | Locally-loaded post-trained checkpoint (`POSTTRAINED_MODEL_PATH` env var) |

Quick start:
```bash
# Baseline — GPT-4o, Python, no tools
python tools/run_large_models.py --id 101248502 --model gpt-4o --language python

# BI-Agent — GPT-4o, Python, full tool suite
python tools/run_large_models.py --id 101248502 --model gpt-4o --language python --tool

# Post-trained BI-Agent
POSTTRAINED_MODEL_PATH=/checkpoints/rft/py_tool \
python tools/run_posttrained.py --id 101248502 --language python --tool
```

See [`tools/README.md`](tools/README.md) for full setup and usage.

---

### `nl2sql/` — NL2SQL Evaluation

Evaluates NL-to-SQL / NL-to-Python models that were trained on the BIBench task. Supports:

- **HuggingFace models** via `run.py` (e.g., `infy-32b`, `xiyan-32b`, or any local checkpoint)
- **Databao agent** via `databao-agent/bi-bench.py` (multi-step GPT-based agent)

```bash
# infy-32b, SQL mode
python nl2sql/run.py \
  --id 101248502 --model infy-32b --language sql \
  --setting baseline --queries-file bi-bench/queries.json --data-dir bi-bench

# xiyan-32b, Python + tool mode
python nl2sql/run.py \
  --id 101248502 --model xiyan-32b --language python --tool \
  --setting tool_eval --queries-file bi-bench/queries.json --data-dir bi-bench
```

See [`nl2sql/README.md`](nl2sql/README.md) for full usage.

---

### `post-train/` — Post-Training Pipeline

End-to-end pipeline for producing post-trained BI-Agent models:

| Stage | Directory | Description |
|---|---|---|
| Data generation | `post-train/data_gen/` | Samples BI cases, generates questions, runs model traces, converts to SFT/RFT training data |
| SFT | `post-train/sft/` | Supervised fine-tuning with MS-Swift |
| RFT | `post-train/rl/` | Reinforcement fine-tuning via GRPO with BIBench reward signal |

```bash
# Generate training data
cd post-train/data_gen
./pipeline_qgen.sh /data/bi_cases 10000
./pipeline_trace.sh

# SFT
cd post-train/sft
bash sft.sh Qwen/Qwen3-8B /data/sft_data /checkpoints/sft

# RFT (GRPO)
cd post-train/rl
bash rft.sh /checkpoints/sft/py_no_tool /data/rft_data py_no_tool
```

See [`post-train/README.md`](post-train/README.md) for full details.

---

## Reproducing Paper Results

### Step 1 — Run BI-Agent baselines and tools (Table 3)

Configure `tools/utils/llm_client.py` with your LLM endpoint, then run each case:

```bash
for id in $(ls bi-bench/ | grep -v 'gt\|queries'); do
  python tools/run_large_models.py --id $id --model gpt-4o --language python --tool \
    --log-csv results_gpt4o_py_tool.csv
done
```

### Step 2 — Post-train models (Table 4)

```bash
# Generate data, run SFT, then RFT as described in post-train/README.md
# Then evaluate the checkpoint:
POSTTRAINED_MODEL_PATH=/checkpoints/rft/py_tool \
python tools/run_posttrained.py --id <id> --language python --tool \
  --log-csv results_posttrained.csv
```

### Step 3 — NL2SQL-system evaluation (Table 4)

```bash
python nl2sql/run.py --id <id> --model infy-32b --language sql \
  --setting eval --queries-file bi-bench/queries.json --data-dir bi-bench
```

---

## Requirements

- Python 3.10+
- PyTorch (for post-trained / HF models)
- `openai`, `azure-ai-inference`, `transformers`, `peft`, `pandas`, `numpy`, `python-dotenv`

LLM access requires an Azure OpenAI or Azure AI Foundry endpoint. See each component's README for environment variable details.
