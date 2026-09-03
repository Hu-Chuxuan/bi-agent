"""
Calculate per-query cost from run log files.

Usage:
    python calc_cost.py --log-csv results.csv --log-dir /datadrive/chuxuan/project/logs \
        --input-price 2.00 --output-price 8.00

Prices are per 1M tokens (defaults match gpt-5.5 tier; adjust as needed).
Outputs a CSV with columns: LogFile, language, tool, case_id,
    input_tokens, output_tokens, cost_usd, AverageSuccessRate
and prints a summary table.
"""
import argparse
import os
import re
import sys
import pandas as pd

import ast

# capture everything between "Raw Responses:\n" and the next "========"
_RAW_RESP_RE = re.compile(
    r"Raw Responses:\s*\n(.*?)(?=\n={4,})",
    re.DOTALL,
)
# extract 'usage': { ... } including one level of nesting
_USAGE_RE = re.compile(r"'usage':\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})")
# direct token fields
_INPUT_TOK_RE  = re.compile(r"'input_tokens':\s*(\d+)")
_OUTPUT_TOK_RE = re.compile(r"'output_tokens':\s*(\d+)")
_CACHED_TOK_RE = re.compile(r"'cached_tokens':\s*(\d+)")

def _parse_usage_block(block: str):
    """Return (input_tokens, output_tokens, cached_tokens) from a Raw Responses block."""
    # try ast.literal_eval on the whole block first
    try:
        obj = ast.literal_eval(block.strip())
        usage = obj.get("usage", {})
        details = usage.get("input_tokens_details", {})
        return (usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                details.get("cached_tokens", 0))
    except Exception:
        pass
    # fallback: regex-extract the usage sub-dict
    m = _USAGE_RE.search(block)
    if m:
        try:
            usage = ast.literal_eval(m.group(1))
            details = usage.get("input_tokens_details", {})
            return (usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    details.get("cached_tokens", 0))
        except Exception:
            pass
    # last resort: pull numbers directly with regex
    inp = int(_INPUT_TOK_RE.search(block).group(1))  if _INPUT_TOK_RE.search(block)  else 0
    out = int(_OUTPUT_TOK_RE.search(block).group(1)) if _OUTPUT_TOK_RE.search(block) else 0
    cac = int(_CACHED_TOK_RE.search(block).group(1)) if _CACHED_TOK_RE.search(block) else 0
    return inp, out, cac

def parse_log_file(path: str):
    """Return (total_input_tokens, total_output_tokens, total_cached_tokens) across all runtimes."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except FileNotFoundError:
        return None, None, None

    input_tok = output_tok = cached_tok = 0
    for m in _RAW_RESP_RE.finditer(text):
        inp, out, cac = _parse_usage_block(m.group(1))
        input_tok  += inp
        output_tok += out
        cached_tok += cac
    return input_tok, output_tok, cached_tok

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-csv",  default="/datadrive/chuxuan/total_logs/0613_gpt-5.5_per_case.csv")
    parser.add_argument("--log-dir", default="/datadrive/chuxuan/project")
    parser.add_argument("--input-price",   type=float, default=5.00,
                        help="Input token price per 1M tokens (USD)")
    parser.add_argument("--cached-price",  type=float, default=2.50,
                        help="Cached input token price per 1M tokens (USD); defaults to half of input-price if not set")
    parser.add_argument("--output-price",  type=float, default=30.00,
                        help="Output token price per 1M tokens (USD)")
    parser.add_argument("--runs",          type=int, default=10,
                        help="Number of runs per case (log files named 0.log … N-1.log)")
    parser.add_argument("--out-csv",       default=None,   help="Optional output CSV path")
    args = parser.parse_args()
    cached_price = args.cached_price if args.cached_price is not None else args.input_price / 2

    df = pd.read_csv(args.log_csv)

    # parse language, tool, case_id from LogFile
    extracted = df["LogFile"].str.extract(r"_([a-z]+)_(True|False)_([^_/]+)_\d{8}")
    df["language"] = extracted[0]
    df["tool"]     = extracted[1].map({"True": True, "False": False})
    df["case_id"]  = extracted[2]

    rows = []
    for _, row in df.iterrows():
        log_folder = os.path.join(args.log_dir, row["LogFile"])
        total_in = total_out = total_cached = 0
        actual_runs = 0
        for i in range(args.runs):
            log_path = os.path.join(log_folder, f"{i}.log")
            inp, out, cac = parse_log_file(log_path)
            if inp is not None:
                total_in     += inp
                total_out    += out
                total_cached += cac
                actual_runs  += 1

        non_cached = total_in - total_cached
        cost = (non_cached * args.input_price + total_cached * cached_price + total_out * args.output_price) / 1_000_000
        cost_per_run = cost / actual_runs if actual_runs > 0 else 0.0

        rows.append({
            "LogFile":          row["LogFile"],
            "case_id":          row["case_id"],
            "language":         row["language"],
            "tool":             row["tool"],
            "input_tokens":     total_in,
            "cached_tokens":    total_cached,
            "output_tokens":    total_out,
            "cost_usd":         round(cost, 6),
            "cost_usd_per_run": round(cost_per_run, 6),
            "AverageSuccessRate": row["AverageSuccessRate"],
        })

    out_df = pd.DataFrame(rows)

    # summary
    summary = out_df.groupby(["language", "tool"]).agg(
        n_cases            = ("case_id",           "nunique"),
        mean_cost_usd      = ("cost_usd",          "mean"),
        total_cost_usd     = ("cost_usd",          "sum"),
        mean_cost_per_run  = ("cost_usd_per_run",  "mean"),
        mean_success       = ("AverageSuccessRate", "mean"),
    ).round(4)
    summary["mean_cost_cents"]         = (summary["mean_cost_usd"]    * 100).round(4)
    summary["mean_cost_cents_per_run"] = (summary["mean_cost_per_run"]* 100).round(4)
    print("\n=== Cost summary ===")
    print(summary[["n_cases","mean_success","mean_cost_cents","mean_cost_usd","total_cost_usd","mean_cost_cents_per_run"]].to_string())

    if args.out_csv:
        out_df.to_csv(args.out_csv, index=False)
        print(f"\nPer-case breakdown saved to {args.out_csv}")


if __name__ == "__main__":
    main()
