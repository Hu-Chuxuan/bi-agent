import glob
import os
import re
import math
import pandas as pd
import json
import argparse
import traceback
import csv
import sqlite3
import uuid
import shutil
import numpy as np

from dotenv import load_dotenv
from utils.prompt import (
    PROMPT_PY_TOOL, TOOLS_PY, PROMPT_SQL_TOOL, TOOLS_SQL,
    DEFAULT_SYSTEM_PROMPT, PROMPT_PY, PROMPT_SQL,
    TOOLS_PY_BASELINE, TOOLS_SQL_BASELINE, FUNCTION_STOPPER,
)
from utils.py_tools import execute_tool as execute_py_tool
from utils.sql_tools_gt import execute_tool as execute_sql_tool
from utils.hf_model import query_chat_endpoint as hf_query, MODEL_REGISTRY
from utils.file2var import sanitize_name
from datetime import datetime
from collections import Counter
from collections.abc import Iterable
from multiprocessing import Process, Queue
from typing import List


class ToolFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class ToolCall:
    def __init__(self, function: ToolFunction):
        self.function = function


def extract_first_json(text):
    start = re.search(r"[\[{]", text)
    if not start:
        return None
    stack = []
    i = start.start()
    for j in range(i, len(text)):
        c = text[j]
        if c in "[{":
            stack.append(c)
        elif c in "]}":
            if not stack:
                return None
            opener = stack.pop()
            if (opener == "[" and c != "]") or (opener == "{" and c != "}"):
                return None
            if not stack:
                return text[i:j+1]
    return None


def parse_tool_call_string_sql(tool_call_string: str) -> List[ToolCall]:
    try:
        match = re.search(r"```json\s*(\[\s*{.*?}\s*]|\{.*?\})\s*```", tool_call_string, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            match = re.search(r"```sql\s+(.*?)```", tool_call_string, re.DOTALL)
            if match:
                code = match.group(1).strip()
                return [ToolCall(function=ToolFunction(name='execute_code', arguments={"code": code}))]
            if tool_call_string.strip().startswith("SELECT"):
                return [ToolCall(function=ToolFunction(name='execute_code', arguments={"code": tool_call_string.strip()}))]
            json_str = extract_first_json(tool_call_string)
            if json_str is None:
                json_str = tool_call_string.strip()

        data = json.loads(json_str)

        if isinstance(data, list) and all(isinstance(s, str) for s in data):
            parsed = []
            for s in data:
                try:
                    parsed.append(json.loads(s))
                except json.JSONDecodeError:
                    continue
            data = parsed

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
                tool_calls.append(ToolCall(function=ToolFunction(name=func_name, arguments=json.dumps(arguments))))
        return tool_calls

    except Exception as e:
        print(f"[parse_tool_call_string_sql] Error: {e}")
        return []


def parse_tool_call_string_python(tool_call_string: str) -> List[ToolCall]:
    try:
        match = re.search(r"```json\s*(\[\s*{.*?}\s*]|\{.*?\})\s*```", tool_call_string, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            match = re.search(r"```python\s+(.*?)```", tool_call_string, re.DOTALL)
            if match:
                code = match.group(1).strip()
                return [ToolCall(function=ToolFunction(name='execute_code', arguments={"code": code}))]
            if tool_call_string.strip().startswith("import"):
                return [ToolCall(function=ToolFunction(name='execute_code', arguments={"code": tool_call_string.strip()}))]
            json_str = extract_first_json(tool_call_string)
            if json_str is None:
                json_str = tool_call_string.strip()

        data = json.loads(json_str)

        try:
            if isinstance(data, str) and re.match(r'^\s*[\[{]', data):
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
                tool_calls.append(ToolCall(function=ToolFunction(name=func_name, arguments=json.dumps(arguments))))
        return tool_calls

    except Exception as e:
        print(f"[parse_tool_call_string_python] Error: {e}")
        return []


def compare_with_timeout(gt_list, result, timeout=600):
    def target(q):
        try:
            res = compare_dataframes_with_cell_sets(gt_list, result)
            q.put(res)
        except Exception:
            q.put(False)

    q = Queue()
    p = Process(target=target, args=(q,))
    p.start()
    p.join(timeout)

    if p.is_alive():
        print("[WARNING] Comparison took too long, terminating.")
        p.terminate()
        p.join()
        return False
    else:
        return q.get()


def parse_value(val):
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('$'):
            try:
                return float(val.replace('$', ''))
            except ValueError:
                return val
        if val.endswith('%'):
            try:
                return float(val.replace('%', '')) / 100
            except ValueError:
                return val
        try:
            return float(val)
        except ValueError:
            return val
    return val


def safe_compare(a, b, rtol=1e-3, atol=0.5) -> bool:
    a = parse_value(a)
    b = parse_value(b)
    try:
        return np.isclose(float(a), float(b), rtol=rtol, atol=atol)
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(a) == pd.to_datetime(b)
        except Exception:
            return str(a).strip() == str(b).strip()


def is_scalar(val):
    return not isinstance(val, Iterable) or isinstance(val, (str, bytes))


def clean_flat_values(vals):
    return [0.0 if is_scalar(v) and pd.isna(v) else v for v in vals]


MISSING_STRINGS = {"", "na", "n/a", "nan", "null", "none", "-", "--"}


def is_missingish(x) -> bool:
    if x is None:
        return True
    if pd.isna(x):
        return True
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


def normalize_df(df, max_rows=100):
    df_sorted = df.copy().sort_values(
        by=list(df.columns), key=lambda col: col.astype(str), kind="mergesort"
    )
    if len(df_sorted) > max_rows:
        df_sorted = df_sorted.iloc[:max_rows]
    return df_sorted.reset_index(drop=True)


def compare_dataframes_with_cell_sets(gt_list, df2) -> bool:
    for df1 in gt_list:
        if df1.shape == (1, 1):
            gt_val = df1.iloc[0, 0]
            for val2 in clean_flat_values(df2.values.flatten().tolist()):
                if safe_compare(gt_val, val2):
                    return True
            continue

        if df1.shape != df2.shape:
            continue
        df1 = normalize_df(df1)
        df2 = normalize_df(df2)
        flat1 = clean_flat_values(df1.values.flatten().tolist())
        flat2 = clean_flat_values(df2.values.flatten().tolist())

        used = [False] * len(flat2)
        all_matched = True

        for val1 in flat1:
            best_idx = -1
            best_score = float("inf")
            for i, val2 in enumerate(flat2):
                if used[i]:
                    continue
                if is_missingish(val1) and is_missingish(val2):
                    best_score = 0
                    best_idx = i
                    break
                try:
                    a, b = float(val1), float(val2)
                    if np.isclose(a, b, rtol=1e-2, atol=0.5):
                        diff = abs(a - b)
                        if diff < best_score:
                            best_score = diff
                            best_idx = i
                except Exception:
                    if safe_compare(val1, val2):
                        best_score = 0
                        best_idx = i
                        break
            if best_idx != -1:
                used[best_idx] = True
            else:
                all_matched = False
                break

        if all_matched and all(used):
            return True

    return False


def create_temp_folder(folder, data_dir):
    unique_id = uuid.uuid4().hex
    src_path = os.path.join(data_dir, folder)
    dst_path = os.path.join(".", "temp", f"{folder}_{unique_id}")

    os.makedirs(dst_path, exist_ok=True)
    if os.path.exists(src_path):
        for root, dirs, files in os.walk(src_path):
            rel_path = os.path.relpath(root, src_path)
            target_dir = os.path.join(dst_path, rel_path)
            os.makedirs(target_dir, exist_ok=True)
            for file in files:
                shutil.copy2(os.path.join(root, file), os.path.join(target_dir, file))

    return dst_path


def main(folder, model_id, language, log_path, tool, queries_file, data_dir):
    load_dotenv()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    temp_folder = create_temp_folder(folder, data_dir)

    with open(queries_file, "r") as f:
        queries = json.load(f)
    query = queries[folder]["query"]

    csv_list = [
        f for f in glob.glob(os.path.join(temp_folder, "*.csv"))
        if not os.path.basename(f).startswith("_")
    ]

    conn = None
    db_path = None
    if language == "sql":
        df_list = {}
        data_list = []
        for _csv in csv_list:
            try:
                table_name = sanitize_name(os.path.basename(_csv)[:-4])
                df = pd.read_csv(_csv, low_memory=False, dtype=str, quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip')
                df.columns = [sanitize_name(col) for col in df.columns]
                data_list.append(table_name)
                df_list[table_name] = df
            except Exception:
                pass

        os.makedirs("sqlite_db", exist_ok=True)
        db_path = f"sqlite_db/sqlite_db_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db"
        conn = sqlite3.connect(db_path)
        for table_name, df in df_list.items():
            df.to_sql(table_name, conn, if_exists='replace', index=False)

    if tool:
        if language == "sql":
            prompt = PROMPT_SQL_TOOL.format(query=query, all_files=data_list)
            system_prompt = DEFAULT_SYSTEM_PROMPT + str(TOOLS_SQL + FUNCTION_STOPPER)
        else:
            prompt = PROMPT_PY_TOOL.format(query=query, all_files=csv_list)
            system_prompt = DEFAULT_SYSTEM_PROMPT + str(TOOLS_PY + FUNCTION_STOPPER)
    else:
        if language == "sql":
            prompt = PROMPT_SQL.format(query=query, all_files=data_list)
            system_prompt = DEFAULT_SYSTEM_PROMPT + str(TOOLS_SQL_BASELINE + FUNCTION_STOPPER)
        else:
            prompt = PROMPT_PY.format(query=query, all_files=csv_list)
            system_prompt = DEFAULT_SYSTEM_PROMPT + str(TOOLS_PY_BASELINE + FUNCTION_STOPPER)

    input_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

    result = None
    max_retries = 10
    i = 0
    tool_call_seq = []
    state = {}

    while i < max_retries:
        i += 1
        try:
            message = hf_query(input_messages, model_id=model_id)
        except Exception:
            message = None
        if message is None:
            break
        print(message)

        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(f"========Runtime {i}======== \n")
                f.write("Input Messages: \n")
                f.write(str(input_messages))
                f.write("\nRaw Responses: \n")
                f.write(message)
                f.write("\n\n")

        tool_calls = parse_tool_call_string_sql(message) if language == "sql" else parse_tool_call_string_python(message)
        stop = False

        if len(tool_calls) > 0:
            input_messages.append({"role": "assistant", "content": message})
            responses = []
            for tool_call in tool_calls:
                if tool_call.function.name == "stopper":
                    stop = True
                    break
                tool_call_seq.append(tool_call.function.name)
                tmp_res = None
                output = ""
                code = ""

                try:
                    if language == "sql":
                        result_dict = execute_sql_tool(tool_call, queries, folder, conn, temp_folder, data_list)
                    else:
                        exec(
                            "import pandas as pd\n"
                            "pd.set_option('display.max_columns', None)\n"
                            "pd.set_option('display.width', None)\n"
                            "pd.set_option('display.max_colwidth', None)",
                            state,
                        )
                        result_dict = execute_py_tool(tool_call, queries, folder, state, temp_folder, csv_list)

                    execution_res = result_dict.get("execution_results", {})
                    output = execution_res.get("outputs", "")
                    if result_dict.get("tool_name") == "execute_code":
                        code = execution_res.get("code", "")
                        tmp_res = execution_res.get("result", None)
                        state = execution_res.get("state", state)
                        if isinstance(tmp_res, pd.DataFrame):
                            tmp_res_str = tmp_res.head().to_string(index=False)
                        else:
                            tmp_res_str = str(tmp_res)
                        res_result = json.dumps({"result (first 5 rows)": tmp_res_str, "outputs": output})
                    else:
                        res_result = json.dumps({"outputs": output})

                except Exception:
                    tb = traceback.format_exc()
                    res_result = json.dumps({"outputs": f"[ERROR] Exception occurred during tool execution:\n{tb}"})

                if tmp_res is not None:
                    result = tmp_res
                responses.append(res_result)

            input_messages.append({"role": "user", "content": f"The responses of {message} is {str(responses)}"})
        else:
            with open(log_path, "a") as f:
                f.write(f"========Processed Info======== \nprompt: \n{prompt}\nresponse: \n{message}\n\n")
            break

        if stop:
            with open(log_path, "a") as f:
                f.write(f"========Processed Info======== \nprompt: \n{prompt}\nresponse: \n{message}\n\n")
            break

        with open(log_path, "a") as f:
            f.write(f"========Processed Info======== \n")
            f.write(f"prompt: \n{prompt}")
            f.write(f"code: {code} \n")
            f.write(f"message: {message} \n")
            f.write(f"output: {output} \n")
            f.write(f"result: {result} \n\n")

    if language == "sql" and conn is not None:
        conn.close()
        if db_path and os.path.exists(db_path):
            os.remove(db_path)

    if result is not None:
        if isinstance(result, (int, float, str)):
            result = pd.DataFrame({"result": [result]})
        gt_dir = os.path.join(data_dir, "gt")
        pattern = os.path.join(gt_dir, f"{folder}_*.csv")
        file_paths = glob.glob(pattern)
        if len(file_paths) == 0:
            gt = [pd.read_csv(os.path.join(gt_dir, f"{folder}.csv"))]
        else:
            gt = [pd.read_csv(fp) for fp in file_paths]
        try:
            final_res = compare_with_timeout(gt, result)
        except Exception:
            final_res = False
    else:
        final_res = False

    with open(log_path, "a") as f:
        f.write("tool call seq:\n")
        f.write(str(tool_call_seq))
        f.write("\npred:\n")
        f.write(result.head().to_string(index=False) if isinstance(result, pd.DataFrame) else str(result))
        f.write("\n\ngt:\n")
        try:
            gt_log = queries[folder]["gt"]
        except KeyError:
            gt_dir = os.path.join(data_dir, "gt")
            pattern = os.path.join(gt_dir, f"{folder}_*.csv")
            file_paths = glob.glob(pattern)
            if len(file_paths) == 0:
                gt_log = [pd.read_csv(os.path.join(gt_dir, f"{folder}.csv")).head().to_string(index=False)]
            else:
                gt_log = [pd.read_csv(fp).head().to_string(index=False) for fp in file_paths]
        f.write(str(gt_log))
        f.write("\n\nfinal result:\n")
        f.write(str(final_res))

    print("[DEBUG]", temp_folder)
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)

    return final_res, tool_call_seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="Query/case ID")
    parser.add_argument("--model", required=True,
                        help=f"Model name (known: {list(MODEL_REGISTRY.keys())}) or HuggingFace model ID / local path")
    parser.add_argument("--language", choices=["python", "sql"], required=True)
    parser.add_argument("--runs", default=1, type=int, help="Attempts per case (stops early on success)")
    parser.add_argument("--tool", action="store_true", help="Use tool-augmented prompts")
    parser.add_argument("--setting", required=True, help="Experiment tag used for log/CSV naming")
    parser.add_argument("--queries-file", required=True,
                        help="Path to JSON file mapping case IDs to queries and ground truth")
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing {id}/ case folders and gt/ subfolder")
    args = parser.parse_args()

    from utils.hf_model import MODEL_REGISTRY as _REG
    model_id = _REG.get(args.model, args.model)

    csv_log_path = f"{args.model}-{args.setting}.csv"
    if os.path.exists(csv_log_path):
        log_df = pd.read_csv(csv_log_path)
        prefix = f"logs/{args.model}-{args.setting}/{args.model}_{args.language}_{args.tool}_{args.id}"
        if log_df["LogFile"].str.startswith(prefix).any():
            print(f"Skipping: Found existing log entry with prefix {prefix}")
            exit(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = f"logs/{args.model}-{args.setting}/{args.model}_{args.language}_{args.tool}_{args.id}_{timestamp}"
    os.makedirs(base_log_dir, exist_ok=True)

    final_res_ls = []
    tool_call_stat = []
    tool_call_order = []

    for i in range(args.runs):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(base_log_dir, f"{i}.log")
        final_res, tool_call_seq = main(
            args.id, model_id, args.language, log_path,
            args.tool, args.queries_file, args.data_dir,
        )
        final_res_ls.append(final_res)
        tool_call_stat += tool_call_seq
        tool_call_order.append(tool_call_seq)

    true_ratio = np.mean(final_res_ls)
    tool_call_summary = json.dumps(Counter(tool_call_stat))
    print(f"Average success rate for {base_log_dir}: {true_ratio:.2%}")

    header = ["LogFile", "AverageSuccessRate", "ToolCallStats", "ToolCallOrders"]
    file_exists = os.path.exists(csv_log_path)
    with open(csv_log_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([base_log_dir, f"{true_ratio:.4f}", tool_call_summary, tool_call_order])
