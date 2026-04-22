import utils.caller_posttrained as caller  # only line that differs from run_large_models.py

import glob
import os
import re
import math
import csv
import json
import uuid
import shutil
import sqlite3
import argparse
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from collections.abc import Iterable
from multiprocessing import Process, Queue
from typing import List

from dotenv import load_dotenv
from utils.prompt import (
    PROMPT_PY_TOOL, PROMPT_SQL_TOOL, PROMPT_PY, PROMPT_SQL,
    TOOLS_PY, TOOLS_SQL, TOOLS_PY_BASELINE, TOOLS_SQL_BASELINE,
    DEFAULT_SYSTEM_PROMPT, FUNCTION_STOPPER,
)
from utils.file2var import sanitize_name

# ---------------------------------------------------------------------------
# Path constants (all relative to this file's location)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BI_BENCH_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "bi-bench")
GT_DIR       = os.path.join(BI_BENCH_DIR, "gt")
QUERIES_FILE = os.path.join(SCRIPT_DIR, "queries.json")
TEMP_DIR     = os.path.join(SCRIPT_DIR, "temp")
SQLITE_DIR   = os.path.join(SCRIPT_DIR, "sqlite_db")

# ---------------------------------------------------------------------------
# Text-based tool-call parsing
# ---------------------------------------------------------------------------

class ToolFunction:
    def __init__(self, name: str, arguments):
        self.name = name
        self.arguments = arguments


class ToolCall:
    def __init__(self, function: ToolFunction):
        self.function = function


def _extract_first_json(text: str):
    start = 0
    while True:
        match = re.search(r"[\[{]", text[start:])
        if not match:
            return None
        i = start + match.start()
        stack = []
        for j in range(i, len(text)):
            c = text[j]
            if c in "[{":
                stack.append(c)
            elif c in "]}":
                if not stack:
                    break
                opener = stack.pop()
                if (opener == "[" and c != "]") or (opener == "{" and c != "}"):
                    break
                if not stack:
                    candidate = text[i : j + 1].strip()
                    try:
                        parsed = json.loads(candidate)
                        if parsed == [] or parsed == {}:
                            start = j + 1
                            break
                        return candidate
                    except Exception:
                        start = j + 1
                        break
        else:
            break
    return None


def _parse_json_to_tool_calls(data) -> List[ToolCall]:
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        return []
    tool_calls = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function" and "name" in item and "parameters" in item:
            func_name = item["name"]
            arguments = item["parameters"]
        elif "function" in item and "arguments" in item:
            func_name = item["function"]
            arguments = item["arguments"]
        else:
            continue
        if isinstance(arguments, (dict, list)):
            tool_calls.append(ToolCall(ToolFunction(func_name, json.dumps(arguments))))
    return tool_calls


def parse_tool_calls_sql(text: str) -> List[ToolCall]:
    try:
        match = re.search(r"```json\s*(\[\s*{.*?}\s*]|\{.*?\})\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            m = re.search(r"```sql\s+(.*?)```", text, re.DOTALL)
            if m:
                return [ToolCall(ToolFunction("execute_code", json.dumps({"code": m.group(1).strip()})))]
            if text.strip().startswith("SELECT"):
                return [ToolCall(ToolFunction("execute_code", json.dumps({"code": text.strip()})))]
            json_str = _extract_first_json(text) or text.strip()
        data = json.loads(json_str)
        if isinstance(data, list) and all(isinstance(s, str) for s in data):
            data = [json.loads(s) for s in data]
        return _parse_json_to_tool_calls(data)
    except Exception as e:
        print(f"[parse_tool_calls_sql] Error: {e}")
        return []


def parse_tool_calls_python(text: str) -> List[ToolCall]:
    try:
        match = re.search(r"```json\s*(\[\s*{.*?}\s*]|\{.*?\})\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            m = re.search(r"```python\s+(.*?)```", text, re.DOTALL)
            if m:
                return [ToolCall(ToolFunction("execute_code", json.dumps({"code": m.group(1).strip()})))]
            if text.strip().startswith("import"):
                return [ToolCall(ToolFunction("execute_code", json.dumps({"code": text.strip()})))]
            json_str = _extract_first_json(text) or text.strip()
        data = json.loads(json_str)
        try:
            if isinstance(data, str) and re.match(r"^\s*[\[{]", data):
                data = json.loads(data)
        except Exception:
            pass
        if isinstance(data, list) and all(isinstance(s, str) for s in data):
            parsed = []
            for s in data:
                try:
                    parsed.append(json.loads(s))
                except json.JSONDecodeError:
                    continue
            data = parsed
        return _parse_json_to_tool_calls(data)
    except Exception as e:
        print(f"[parse_tool_calls_python] Error: {e}")
        return []


# ---------------------------------------------------------------------------
# DataFrame comparison helpers
# ---------------------------------------------------------------------------

MISSING_STRINGS = {"", "na", "n/a", "nan", "null", "none", "-", "--"}


def _parse_value(val):
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("$"):
            try:
                return float(val.replace("$", ""))
            except ValueError:
                return val
        if val.endswith("%"):
            try:
                return float(val.replace("%", "")) / 100
            except ValueError:
                return val
        try:
            return float(val)
        except ValueError:
            return val
    return val


def _safe_compare(a, b, rtol=1e-3, atol=0.5) -> bool:
    a, b = _parse_value(a), _parse_value(b)
    try:
        return bool(np.isclose(float(a), float(b), rtol=rtol, atol=atol))
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(a) == pd.to_datetime(b)
        except Exception:
            return str(a).strip() == str(b).strip()


def _is_scalar(val):
    return not isinstance(val, Iterable) or isinstance(val, (str, bytes))


def _is_missingish(x) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and x.strip().lower() in MISSING_STRINGS:
        return True
    try:
        if math.isinf(float(x)):
            return True
    except Exception:
        pass
    return False


def _clean_flat(vals):
    return [0.0 if (_is_scalar(v) and _is_missingish(v)) else v for v in vals]


def _normalize_df(df, max_rows=100):
    df = df.copy().sort_values(
        by=list(df.columns), key=lambda col: col.astype(str), kind="mergesort"
    )
    return df.iloc[:max_rows].reset_index(drop=True)


def compare_dataframes(gt_list, df2) -> bool:
    for df1 in gt_list:
        if df1.shape == (1, 1):
            gt_val = df1.iloc[0, 0]
            for v2 in _clean_flat(df2.values.flatten().tolist()):
                if _safe_compare(gt_val, v2):
                    return True
            continue
        if df1.shape != df2.shape:
            continue
        flat1 = _clean_flat(_normalize_df(df1).values.flatten().tolist())
        flat2 = _clean_flat(_normalize_df(df2).values.flatten().tolist())
        used = [False] * len(flat2)
        all_matched = True
        for v1 in flat1:
            best_idx, best_score = -1, float("inf")
            for i, v2 in enumerate(flat2):
                if used[i]:
                    continue
                if _is_missingish(v1) and _is_missingish(v2):
                    best_idx, best_score = i, 0
                    break
                try:
                    a, b = float(v1), float(v2)
                    if np.isclose(a, b, rtol=1e-2, atol=0.5):
                        diff = abs(a - b)
                        if diff < best_score:
                            best_score, best_idx = diff, i
                except Exception:
                    if _safe_compare(v1, v2):
                        best_idx, best_score = i, 0
                        break
            if best_idx != -1:
                used[best_idx] = True
            else:
                all_matched = False
                break
        if all_matched and all(used):
            return True
    return False


def compare_with_timeout(gt_list, result, timeout=600) -> bool:
    def _target(q):
        try:
            q.put(compare_dataframes(gt_list, result))
        except Exception:
            q.put(False)

    q = Queue()
    p = Process(target=_target, args=(q,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        print("[WARNING] Comparison timed out.")
        p.terminate()
        p.join()
        return False
    return q.get()


# ---------------------------------------------------------------------------
# Temp-folder management
# ---------------------------------------------------------------------------

def create_temp_folder(folder_id: str) -> str:
    unique_id = uuid.uuid4().hex
    src = os.path.join(BI_BENCH_DIR, folder_id)
    dst = os.path.join(TEMP_DIR, f"{folder_id}_{unique_id}")
    os.makedirs(dst, exist_ok=True)
    if os.path.exists(src):
        for root, dirs, files in os.walk(src):
            rel = os.path.relpath(root, src)
            target = os.path.join(dst, rel)
            os.makedirs(target, exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(root, f), os.path.join(target, f))
    return dst


# ---------------------------------------------------------------------------
# Single tool-call execution
# ---------------------------------------------------------------------------

def _run_tool_call(tc, language, queries, folder_id, conn, temp_folder, csv_list, data_list, state):
    tmp_res = None
    output = ""
    code = ""
    try:
        if language == "sql":
            result_dict = caller.execute_sql_tool(tc, queries, folder_id, conn, temp_folder, data_list)
        else:
            exec(
                "import pandas as pd\n"
                "pd.set_option('display.max_columns', None)\n"
                "pd.set_option('display.width', None)\n"
                "pd.set_option('display.max_colwidth', None)",
                state,
            )
            result_dict = caller.execute_py_tool(tc, queries, folder_id, state, temp_folder, csv_list)

        exec_res = result_dict.get("execution_results", {})
        output = exec_res.get("outputs", "")
        if result_dict.get("tool_name") == "execute_code":
            code = exec_res.get("code", "")
            tmp_res = exec_res.get("result", None)
            state = exec_res.get("state", state)
            tmp_res_str = tmp_res.head().to_string(index=False) if isinstance(tmp_res, pd.DataFrame) else str(tmp_res)
            res_result = json.dumps({"result (first 5 rows)": tmp_res_str, "outputs": output})
        else:
            res_result = json.dumps({"outputs": output})
    except Exception:
        res_result = json.dumps({"outputs": f"[ERROR] Exception during tool execution:\n{traceback.format_exc()}"})

    return tmp_res, code, output, state, res_result


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def main(folder_id: str, model: str, language: str, tool: bool, log_path: str):
    load_dotenv()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    temp_folder = create_temp_folder(folder_id)

    with open(QUERIES_FILE) as f:
        queries = json.load(f)
    query = queries[folder_id]

    csv_list = [
        fp for fp in glob.glob(os.path.join(temp_folder, "*.csv"))
        if not os.path.basename(fp).startswith("_")
    ]

    conn = None
    db_path = None
    data_list = []

    if language == "sql":
        df_list = {}
        for _csv in csv_list:
            try:
                tname = sanitize_name(os.path.basename(_csv)[:-4])
                df = pd.read_csv(_csv, low_memory=False, dtype=str,
                                 quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
                df.columns = [sanitize_name(c) for c in df.columns]
                data_list.append(tname)
                df_list[tname] = df
            except Exception:
                pass
        os.makedirs(SQLITE_DIR, exist_ok=True)
        db_path = os.path.join(SQLITE_DIR, f"db_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db")
        conn = sqlite3.connect(db_path)
        for tname, df in df_list.items():
            df.to_sql(tname, conn, if_exists="replace", index=False)

    if language == "sql":
        tools_spec = TOOLS_SQL if tool else TOOLS_SQL_BASELINE
        prompt = (PROMPT_SQL_TOOL if tool else PROMPT_SQL).format(query=query, all_files=data_list)
    else:
        tools_spec = TOOLS_PY if tool else TOOLS_PY_BASELINE
        prompt = (PROMPT_PY_TOOL if tool else PROMPT_PY).format(query=query, all_files=csv_list)

    system_prompt = DEFAULT_SYSTEM_PROMPT + str(tools_spec + FUNCTION_STOPPER)
    input_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

    result = None
    tool_call_seq = []
    state = {}

    for i in range(1, 31):
        response = caller.call_model(model, input_messages, tools_spec)

        with open(log_path, "a") as lf:
            lf.write(f"========Runtime {i}========\n")
            lf.write(f"Input Messages:\n{input_messages}\nRaw Response:\n{response}\n\n")

        if response is None:
            break

        message_text = response
        print(message_text)
        parse_fn = parse_tool_calls_sql if language == "sql" else parse_tool_calls_python
        tool_calls = parse_fn(message_text)

        if not tool_calls:
            with open(log_path, "a") as lf:
                lf.write(f"========Processed Info========\nprompt:\n{prompt}\nresponse:\n{message_text}\n\n")
            break

        input_messages.append(caller.format_assistant_turn(response, message_text))
        responses = []
        stop = False

        for tc in tool_calls:
            if tc.function.name == "stopper":
                stop = True
                break
            tool_call_seq.append(tc.function.name)
            tmp_res, code, output, state, res_result = _run_tool_call(
                tc, language, queries, folder_id, conn, temp_folder, csv_list, data_list, state
            )
            if tmp_res is not None:
                result = tmp_res
            responses.append(res_result)

        input_messages.append(caller.format_tool_turn(message_text, responses))

        with open(log_path, "a") as lf:
            lf.write(f"========Processed Info========\nprompt:\n{prompt}\nresponse:\n{message_text}\nresult: {result}\n\n")

        if stop:
            break

    if conn is not None:
        conn.close()
    if db_path and os.path.exists(db_path):
        os.remove(db_path)

    final_res = False
    if result is not None:
        if isinstance(result, (int, float, str)):
            result = pd.DataFrame({"result": [result]})
        gt_files = glob.glob(os.path.join(GT_DIR, f"{folder_id}_*.csv"))
        if not gt_files:
            gt_files = [os.path.join(GT_DIR, f"{folder_id}.csv")]
        gt = [pd.read_csv(fp) for fp in gt_files if os.path.exists(fp)]
        if gt:
            try:
                final_res = compare_with_timeout(gt, result)
            except Exception:
                final_res = False

    with open(log_path, "a") as lf:
        lf.write(f"tool call seq:\n{tool_call_seq}\npred:\n")
        lf.write(result.head().to_string(index=False) if isinstance(result, pd.DataFrame) else str(result))
        lf.write("\n\ngt:\n")
        gt_files = glob.glob(os.path.join(GT_DIR, f"{folder_id}_*.csv"))
        if not gt_files:
            gt_files = [os.path.join(GT_DIR, f"{folder_id}.csv")]
        lf.write(str([pd.read_csv(fp).head().to_string(index=False) for fp in gt_files if os.path.exists(fp)]))
        lf.write(f"\n\nfinal result:\n{final_res}")

    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)

    return final_res, tool_call_seq


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="Task ID (folder name in bi-bench)")
    parser.add_argument("--language", choices=["python", "sql"], required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--tool", action="store_true", help="Use full tool set")
    parser.add_argument("--log-csv", default="results_posttrained.csv", help="CSV file to append summary results")
    parser.add_argument("--log-dir", default="logs", help="Base directory for log files")
    args = parser.parse_args()

    model = os.environ.get("POSTTRAINED_MODEL_PATH", "posttrained")
    run_tag = f"posttrained_{args.language}_{args.tool}_{args.id}"
    csv_log_path = args.log_csv

    if os.path.exists(csv_log_path):
        log_df = pd.read_csv(csv_log_path)
        prefix = f"{args.log_dir}/{run_tag}"
        if log_df["LogFile"].str.startswith(prefix).any():
            print(f"Skipping: existing log entry with prefix {prefix}")
            exit(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.join(args.log_dir, f"{run_tag}_{timestamp}")
    os.makedirs(base_log_dir, exist_ok=True)

    final_res_ls, tool_call_stat, tool_call_order = [], [], []

    for i in range(args.runs):
        log_path = os.path.join(base_log_dir, f"{i}.log")
        final_res, tool_call_seq = main(args.id, model, args.language, args.tool, log_path)
        final_res_ls.append(final_res)
        tool_call_stat += tool_call_seq
        tool_call_order.append(tool_call_seq)

    true_ratio = float(np.mean(final_res_ls))
    tool_call_summary = json.dumps(Counter(tool_call_stat))
    print(f"Average success rate for {base_log_dir}: {true_ratio:.2%}")

    file_exists = os.path.exists(csv_log_path)
    with open(csv_log_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["LogFile", "AverageSuccessRate", "ToolCallStats", "ToolCallOrders"])
        writer.writerow([base_log_dir, f"{true_ratio:.4f}", tool_call_summary, tool_call_order])
