import json
import argparse
from pathlib import Path
from typing import List, Dict

def get_jsonl(transformed_traces: str, output_path: str | None = None) -> str:
    """
    Build an SFT‑ready JSONL file where each file contains the full conversation trace,
    but each message only keeps 'role' and 'content'.

    Parameters
    ----------
    transformed_traces : str
        Path to the transformed trace produced earlier.
    output_path : str | None
        Where to save the JSONL file. If None, writes next to the source
        with `_sft.jsonl` appended.

    Returns
    -------
    str
        The path of the JSONL file written to disk.
    """
    # -------- load the full conversation ----------
    with open(transformed_traces, "r", encoding="utf-8") as f:
        raw_traces: List[Dict] = json.load(f)

    # -------- filter only 'role' and 'content' ----------
    cleaned_trace = [{"role": m["role"], "content": m["content"]}
                     for m in raw_traces if "role" in m and "content" in m]

    # -------- create JSONL-compatible list ----------
    jsonl_docs = [{"messages": cleaned_trace}]

    # -------- decide output path ----------
    if output_path is None:
        src = Path(transformed_traces)
        suffix = src.parent.name.removeprefix("transformed_traces")
        output_dir = src.parent.parent / f"jsonl_collection{suffix}"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{src.stem}.jsonl"

    # -------- write JSONL ----------
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in jsonl_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Created 1 SFT example ➜ {output_path}")
    return str(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", required=True, help="Path to transformed trace file")
    args = parser.parse_args()

    get_jsonl(args.filename)