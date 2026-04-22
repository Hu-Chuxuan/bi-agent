# Post-Training

Two-stage pipeline: **SFT** (supervised fine-tuning) followed by **RFT** (reinforcement fine-tuning via GRPO).

Training data for both stages is produced by the generation pipeline in [`data_gen/`](data_gen/README.md). Run that pipeline first to produce the data root used by the scripts below.

## Data

Both stages expect data organised under a single root directory with one sub-folder per mode:

```
<data_root>/
  py_no_tool/
    training_case_ids.jsonl   # Swift-format training samples
    training_case_ids.json    # Case metadata used by the GRPO trainer
    gt/                       # Ground-truth CSVs for reward computation
  py_tool/
    ...
  sql_no_tool/
    ...
  sql_tool/
    ...
```

Download the dataset and place it in this layout before running either script.

## Modes

| Mode | Language | Tools |
|---|---|---|
| `py_no_tool` | Python | none |
| `py_tool` | Python | transform / retrieve / join |
| `sql_no_tool` | SQL | none |
| `sql_tool` | SQL | transform / retrieve / join |

---

## Stage 1 — SFT

```bash
cd sft
bash sft.sh <model> <sft_data_dir> <output_dir> [mode]
```

| Argument | Description |
|---|---|
| `model` | HuggingFace model ID or local path (e.g. `Qwen/Qwen3-8B`) |
| `sft_data_dir` | Path to the data root described above |
| `output_dir` | Base directory for saving LoRA checkpoints |
| `mode` | One of the four modes, or `all` to run all sequentially (default: `all`) |

**Example — train all modes:**
```bash
bash sft.sh Qwen/Qwen3-8B /data/sft_data /checkpoints/sft
```

**Example — train one mode:**
```bash
bash sft.sh Qwen/Qwen3-8B /data/sft_data /checkpoints/sft py_no_tool
```

---

## Stage 2 — RFT (GRPO)

```bash
cd rl
bash rft.sh <model_path> <rft_data_dir> [mode]
```

| Argument | Description |
|---|---|
| `model_path` | Local path to the SFT checkpoint to start from |
| `rft_data_dir` | Path to the data root described above |
| `mode` | One of the four modes (default: `py_no_tool`) |

**Example:**
```bash
bash rft.sh /checkpoints/sft/py_no_tool /data/rft_data py_no_tool
```

The RFT stage runs one mode at a time. Run the script once per mode as needed.
