#!/usr/bin/env python3
"""
Generic agent harness for running *external* BI agents (ktx, Databao Agent, ...)
on BI-Bench and grading their outputs with the same protocol as ``run.py``.

These agents are **not** vendored in this repository — install each one from its
upstream project first (see ``README.md`` → "Other Baselines"), then run it here.
The harness loads a case's tables, builds a SQLite database, invokes the agent,
grades the result against the ground truth, and appends one summary row per case.

Usage:
    python agent_harness.py --agent ktx --id 101248502 \\
        --queries-file ../bi-bench/queries.json --data-dir ../bi-bench \\
        --log-csv ../results/ktx.csv [--runs 1]

Grading mirrors the paper's protocol (Spider2-style numeric tolerance and
missing-value canonicalization; row/column-permutation-invariant cell match).
The reference implementation lives in ``run.py``.
"""
import argparse
import csv as _csv
import glob
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

from utils.file2var import sanitize_name


# --------------------------------------------------------------------------- #
# Case loading                                                                 #
# --------------------------------------------------------------------------- #
def load_query(queries_file: str, case_id: str) -> str:
    with open(queries_file) as f:
        queries = json.load(f)
    return queries[case_id]["query"]


def build_case(case_id: str, data_dir: str):
    """Materialize one case: copy its CSVs to a temp dir and load them into a
    fresh SQLite db (all columns as text, names sanitized). Returns
    ``(csv_dir, db_path, table_names)``. Caller is responsible for cleanup."""
    src = os.path.join(data_dir, str(case_id))
    csvs = [f for f in glob.glob(os.path.join(src, "*.csv"))
            if not os.path.basename(f).startswith("_")]
    if not csvs:
        raise FileNotFoundError(f"no CSV tables for case {case_id} under {src}")

    workdir = tempfile.mkdtemp(prefix=f"bibench_{case_id}_")
    db_path = os.path.join(workdir, "bench.db")
    conn = sqlite3.connect(db_path)
    tables = []
    for c in csvs:
        try:
            df = pd.read_csv(c, low_memory=False, dtype=str,
                             quoting=_csv.QUOTE_MINIMAL, on_bad_lines="skip")
        except Exception:  # noqa: BLE001
            continue
        t = sanitize_name(os.path.basename(c)[:-4])
        df.columns = [sanitize_name(x) for x in df.columns]
        df.to_sql(t, conn, if_exists="replace", index=False)
        shutil.copy(c, os.path.join(workdir, os.path.basename(c)))
        tables.append(t)
    conn.close()
    return workdir, db_path, tables


# --------------------------------------------------------------------------- #
# Grading (mirrors run.py; Spider2-style tolerance)                            #
# --------------------------------------------------------------------------- #
_MISSING = {"", "none", "nan", "null", "n/a", "na"}


def _norm(x):
    if x is None:
        return None
    s = str(x).strip()
    if s.lower() in _MISSING:
        return None
    s2 = s.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return round(float(s2), 2)
    except ValueError:
        return s.lower()


def _rows_multiset(df: pd.DataFrame):
    return sorted(tuple(sorted((str(_norm(v)) for v in row)))
                  for row in df.itertuples(index=False, name=None))


def _match(pred: pd.DataFrame, gt: pd.DataFrame) -> bool:
    if gt.shape == (1, 1):
        target = _norm(gt.iat[0, 0])
        return any(_norm(v) == target for v in pred.to_numpy().ravel())
    if pred.shape != gt.shape:
        return False
    return _rows_multiset(pred) == _rows_multiset(gt)


def grade(pred: pd.DataFrame, data_dir: str, case_id: str) -> bool:
    if pred is None or len(pred) == 0:
        return False
    gt_dir = os.path.join(data_dir, "gt")
    cands = ([os.path.join(gt_dir, f"{case_id}.csv")]
             + sorted(glob.glob(os.path.join(gt_dir, f"{case_id}_*.csv"))))
    for g in cands:
        if not os.path.exists(g):
            continue
        try:
            gt = pd.read_csv(g, dtype=str, keep_default_na=False)
        except Exception:  # noqa: BLE001
            continue
        if _match(pred, gt):
            return True
    return False


# --------------------------------------------------------------------------- #
# LLM client (OpenAI-compatible; configure via env)                            #
# --------------------------------------------------------------------------- #
def _chat(model, messages, tools):
    """One chat-completions call with tool use. Reads OPENAI_API_KEY (and
    optionally OPENAI_BASE_URL) from the environment; swap for your Azure/other
    client here if needed."""
    from openai import OpenAI
    client = OpenAI()
    return client.chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice="auto")


# --------------------------------------------------------------------------- #
# Agent adapters                                                               #
# --------------------------------------------------------------------------- #
class Agent:
    name = "base"

    def __init__(self, model):
        self.model = model

    def run(self, query: str, csv_dir: str, db_path: str, tables) -> pd.DataFrame:
        raise NotImplementedError


class KtxAgent(Agent):
    """ktx (https://github.com/Kaelio/ktx) as the context layer.

    Requires the ``ktx`` CLI on PATH (install per the README).
    """
    name = "ktx"
    semantic_model = "gpt-5.5"
    max_turns = 30

    _TOOLS = [
        {"type": "function", "function": {
            "name": "execute_code",
            "description": "Run a read-only SQLite query against the database through ktx and "
                           "return rows. Use LIMIT 5 while inspecting; the final answer query "
                           "should return the full result.",
            "parameters": {"type": "object",
                           "properties": {"code": {"type": "string", "description": "SQLite SQL."}},
                           "required": ["code"], "additionalProperties": False}}},
        {"type": "function", "function": {
            "name": "search_semantic_layer",
            "description": "Search ktx's semantic layer for tables/columns relevant to a phrase, "
                           "with join relationships and generated descriptions.",
            "parameters": {"type": "object",
                           "properties": {"query": {"type": "string"}},
                           "required": ["query"], "additionalProperties": False}}},
    ]
    _PROMPT = (
        "You are a data analyst. Answer the query with SQLite SQL.\nQuery: {query}\n\n"
        "The data is in a SQLite database exposed through ktx. Tools:\n"
        "- search_semantic_layer: find relevant tables/columns instead of guessing.\n"
        "- execute_code: run read-only SQL and see rows.\n\n"
        "All tables: {tables}\n\n"
        "Search the semantic layer first, inspect with LIMIT 5, then build the answer. "
        "Cast text columns to numeric before arithmetic (every column is stored as text). "
        "Make the final answer the LAST SQL query you run (no LIMIT), then stop calling tools.")

    def _env(self):
        env = dict(os.environ)
        for k in list(env):            # isomorphic-git refuses to run with these set
            if k.startswith("GIT_CONFIG"):
                del env[k]
        return env

    def _ktx(self, pdir, *args, timeout=900):
        p = subprocess.run(["ktx", *args], cwd=pdir, env=self._env(),
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or ""), (p.stderr or "")

    def _write_yaml(self, pdir, db_path):
        m = self.semantic_model
        with open(os.path.join(pdir, "ktx.yaml"), "w") as f:
            f.write(
                "connections:\n  bench:\n    driver: sqlite\n    path: %s\n"
                "llm:\n  provider:\n    backend: codex\n  models:\n"
                "    default: %s\n    triage: %s\n    candidateExtraction: %s\n"
                "    curator: %s\n    reconcile: %s\n    repair: %s\n"
                "scan:\n  enrichment:\n    mode: llm\n    embeddings:\n"
                "      backend: sentence-transformers\n      model: all-MiniLM-L6-v2\n"
                "      dimensions: 384\n  relationships:\n    enabled: true\n    llmProposals: true\n"
                "ingest:\n  embeddings:\n    backend: sentence-transformers\n"
                "    model: all-MiniLM-L6-v2\n    dimensions: 384\n"
                % (db_path, m, m, m, m, m, m))

    def _run_sql(self, pdir, code):
        rc, out, err = self._ktx(pdir, "sql", "-c", "bench", "--json",
                                 "--max-rows", "5000", code, timeout=300)
        body = out[out.find("{"):] if "{" in out else ""
        if rc != 0 or not body:
            return None, "\n".join((err or out).strip().splitlines()[-4:])[:1500] or "query failed"
        try:
            d = json.loads(body)
        except json.JSONDecodeError:
            return None, out[-800:]
        return pd.DataFrame(d.get("rows") or [], columns=d.get("headers") or []), None

    def run(self, query, csv_dir, db_path, tables):
        pdir = os.path.dirname(db_path)
        self._write_yaml(pdir, db_path)
        rc, _, err = self._ktx(pdir, "ingest", "bench", "--no-query-history", timeout=1800)
        if rc != 0:
            raise RuntimeError(f"ktx ingest failed: {err[-500:]}")

        messages = [{"role": "user",
                     "content": self._PROMPT.format(query=query, tables=tables)}]
        last_df = None
        for _ in range(self.max_turns):
            resp = _chat(self.model, messages, self._TOOLS)
            msg = resp.choices[0].message
            messages.append(msg.model_dump())
            if not msg.tool_calls:
                break
            for tc in msg.tool_calls:
                try:
                    a = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    a = {}
                if tc.function.name == "execute_code":
                    df, e = self._run_sql(pdir, a.get("code", ""))
                    if df is not None:
                        last_df = df
                        content = f"{len(df)} rows; columns={list(df.columns)}\n{df.head(20).to_string(index=False)}"
                    else:
                        content = f"ERROR: {e}"
                else:
                    _, so, se = self._ktx(pdir, "sl", a.get("query", ""))
                    content = (so or se)[-2000:]
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": str(content)})
        return last_df


class DatabaoAgent(Agent):
    """Databao Agent — install its upstream package (which ships its own
    ``bi-bench.py`` full-benchmark runner). Because that runner grades cases
    internally and writes its own results CSV, this adapter shells out to it and
    reads the predicted table back.

    Point ``DATABAO_BIBENCH`` at the installed ``bi-bench.py`` (and adjust the
    flags below if your databao version differs)."""
    name = "databao"

    def run(self, query, csv_dir, db_path, tables):
        bibench = os.environ.get("DATABAO_BIBENCH")
        if not bibench or not os.path.exists(bibench):
            raise RuntimeError(
                "Set DATABAO_BIBENCH to the installed databao bi-bench.py. "
                "Install the Databao Agent per its upstream repo first. Note: databao "
                "ships a complete benchmark runner — you can also run it directly instead "
                "of through this harness (see README → Other Baselines).")
        out_dir = tempfile.mkdtemp(prefix="databao_")
        qfile = os.path.join(out_dir, "q.json")
        rfile = os.path.join(out_dir, "result.csv")
        with open(qfile, "w") as f:
            json.dump({"__query__": {"query": query}}, f)
        # data-dir must expose this case's tables + gt; reuse csv_dir (tables only)
        subprocess.run(
            ["python", bibench, "--queries-file", qfile, "--data-dir", csv_dir,
             "--model", self.model, "--runs", "1", "--result-csv", rfile],
            check=True, timeout=1800)
        pred_path = os.path.join(out_dir, "prediction.csv")
        if os.path.exists(pred_path):
            return pd.read_csv(pred_path, dtype=str, keep_default_na=False)
        raise RuntimeError(
            "databao run produced no prediction table at "
            f"{pred_path}; adapt DatabaoAgent.run to your databao version's output.")


AGENTS = {a.name: a for a in (KtxAgent, DatabaoAgent)}


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, choices=sorted(AGENTS))
    ap.add_argument("--id", required=True, help="Case ID (folder under --data-dir).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--queries-file", required=True)
    ap.add_argument("--data-dir", required=True,
                    help="Dir with {id}/ case folders and a gt/ subfolder.")
    ap.add_argument("--runs", type=int, default=1,
                    help="Attempts per case; stops early on first success.")
    ap.add_argument("--log-csv", default=None)
    args = ap.parse_args()

    agent = AGENTS[args.agent](model=args.model)
    query = load_query(args.queries_file, args.id)
    csv_dir, db_path, tables = build_case(args.id, args.data_dir)

    successes = 0
    try:
        for _ in range(args.runs):
            pred = agent.run(query, csv_dir, db_path, tables)
            if grade(pred, args.data_dir, args.id):
                successes += 1
                break
    finally:
        shutil.rmtree(csv_dir, ignore_errors=True)

    rate = successes / max(args.runs, 1)
    print(f"[{args.agent}] case {args.id}: success_rate={rate:.3f}")
    if args.log_csv:
        new = not os.path.exists(args.log_csv)
        with open(args.log_csv, "a", newline="") as f:
            w = _csv.writer(f)
            if new:
                w.writerow(["case_id", "agent", "model", "success_rate", "timestamp"])
            w.writerow([args.id, args.agent, args.model, f"{rate:.3f}",
                        datetime.now().isoformat(timespec="seconds")])


if __name__ == "__main__":
    main()
