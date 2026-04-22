import json
import argparse
from pathlib import Path
from typing import List, Dict


def get_jsonl(
    transformed_traces: str,
    transformed_dir: str = "transformed_traces",
    output_dir: str = "sft_data",
    output_path: str | None = None,
) -> str:
    """
    Build an SFT-ready JSONL file where each line is one training sample
    containing the conversation up to and including one assistant turn.

    Parameters
    ----------
    transformed_traces : str
        Path to the transformed trace produced by transform_trace.py.
    transformed_dir : str
        Name of the transformed traces root folder used to derive output path.
    output_dir : str
        Name of the output sft_data root folder.
    output_path : str | None
        Override output path. If None, mirrors the input structure under output_dir.
    """
    with open(transformed_traces, "r", encoding="utf-8") as f:
        _raw_traces: List[Dict] = json.load(f)

    raw_traces = [
        {"role": m["role"], "content": m["content"]}
        for m in _raw_traces
        if "role" in m and "content" in m
    ]

    jsonl_docs: List[Dict] = []
    for idx, msg in enumerate(raw_traces):
        if msg.get("role") == "assistant":
            jsonl_docs.append({"messages": raw_traces[: idx + 1]})

    if output_path is None:
        src = Path(transformed_traces).resolve()
        try:
            transformed_idx = src.parts.index(transformed_dir)
        except ValueError:
            raise ValueError(f"'{transformed_dir}' not found in path: {src}")

        relative_subpath = Path(*src.parts[transformed_idx + 1:])
        new_root = Path(*src.parts[:transformed_idx]) / output_dir
        output_path = (new_root / relative_subpath).with_suffix(".jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in jsonl_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Created {len(jsonl_docs)} SFT examples -> {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", required=True, help="Path to transformed trace JSON file")
    parser.add_argument("--transformed-dir", default="transformed_traces",
                        help="Name of the transformed traces folder in the path")
    parser.add_argument("--output-dir", default="sft_data",
                        help="Name of the output sft_data folder")
    args = parser.parse_args()

    get_jsonl(args.filename, args.transformed_dir, args.output_dir)
