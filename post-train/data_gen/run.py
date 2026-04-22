import glob
import os
import re
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
from utils.eval_prompt import (
    PROMPT_PY, TOOLS_PY_BASELINE, PROMPT_SQL, TOOLS_SQL_BASELINE,
    PROMPT_PY_TOOL, TOOLS_PY, PROMPT_SQL_TOOL, TOOLS_SQL,
)
from utils.py_tools_gt import execute_tool as execute_py_tool
from utils.sql_tools_gt import execute_tool as execute_sql_tool
from utils.llm_client import query_chat_endpoint
from utils.file2var import sanitize_name
from datetime import datetime
from pathlib import Path

import numpy as np
from collections.abc import Iterable
from multiprocessing import Process, Queue


def compare_with_timeout(gt_list, result, timeout=30):
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
            a_dt = pd.to_datetime(a)
            b_dt = pd.to_datetime(b)
            return a_dt == b_dt
        except Exception:
            return str(a).strip() == str(b).strip()


def is_scalar(val):
    return not isinstance(val, Iterable) or isinstance(val, (str, bytes))


def clean_flat_values(vals):
    cleaned = []
    for v in vals:
        if is_scalar(v):
            cleaned.append(0.0 if pd.isna(v) else v)
        else:
            cleaned.append(v)
    return cleaned


def compare_dataframes_with_cell_sets(gt_list, df2) -> bool:
    for idx, df1 in enumerate(gt_list):
        if df1.shape == (1, 1):
            gt_val = df1.iloc[0, 0]
            flat2 = clean_flat_values(df2.values.flatten().tolist())
            for val2 in flat2:
                if safe_compare(gt_val, val2):
                    return True
            continue

        if df1.shape != df2.shape:
            continue

        flat1 = clean_flat_values(df1.values.flatten().tolist())
        flat2 = clean_flat_values(df2.values.flatten().tolist())

        used = [False] * len(flat2)
        unmatched_vals = []

        for val1 in flat1:
            best_idx = -1
            best_score = float("inf")

            for i, val2 in enumerate(flat2):
                if used[i]:
                    continue
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
                unmatched_vals.append(val1)

        if all(used):
            return True

    return False


def create_temp_folder(folder, transform_method, data_dir):
    unique_id = uuid.uuid4().hex
    folder_name = f"{folder}_{unique_id}"
    src_path = os.path.join(data_dir, f"cases_{transform_method}", folder)
    dst_path = os.path.join(".", "temp", folder_name)

    os.makedirs(dst_path, exist_ok=True)

    if os.path.exists(src_path):
        for root, dirs, files in os.walk(src_path):
            rel_path = os.path.relpath(root, src_path)
            target_dir = os.path.join(dst_path, rel_path)
            os.makedirs(target_dir, exist_ok=True)
            for file in files:
                shutil.copy2(os.path.join(root, file), os.path.join(target_dir, file))

    return dst_path


def main(folder, model, language, log_path, transform_method, use_tool, data_dir, output_dir):
    load_dotenv()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    if use_tool:
        py_prompt, py_tools = PROMPT_PY_TOOL, TOOLS_PY
        sql_prompt, sql_tools = PROMPT_SQL_TOOL, TOOLS_SQL
        mode_suffix = "tool"
    else:
        py_prompt, py_tools = PROMPT_PY, TOOLS_PY_BASELINE
        sql_prompt, sql_tools = PROMPT_SQL, TOOLS_SQL_BASELINE
        mode_suffix = "no_tool"

    temp_folder = create_temp_folder(folder, transform_method, data_dir)

    cases_json = os.path.join(data_dir, f"training_cases_{transform_method}.json")
    with open(cases_json, "r") as f:
        queries = json.load(f)
    query = queries[folder]["query"]

    csv_list = [
        f for f in glob.glob(os.path.join(temp_folder, "*.csv"))
        if not os.path.basename(f).startswith("_")
    ]

    print(csv_list)

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

    if language == "sql":
        prompt = sql_prompt.format(query=query, all_files=data_list)
    else:
        prompt = py_prompt.format(query=query, all_files=csv_list)

    input_messages = [{"role": "user", "content": prompt}]

    result = None
    max_retries = 30
    i = 0
    tool_call_seq = []
    state = {}

    while i < max_retries:
        i += 1
        active_tools = sql_tools if language == "sql" else py_tools

        response = query_chat_endpoint(model, input_messages, active_tools)

        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(f"========Runtime {i}======== \n")
                f.write("Input Messages: \n")
                f.write(str(input_messages))
                f.write("\nRaw Responses: \n")
                f.write(str(response.model_dump()))
                f.write("\n\n")

        message = response.choices[0].message
        print(message)

        input_messages.append(message)

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_call_seq.append(tool_call.function.name)
                tmp_res = None
                output = ""
                code = ""

                try:
                    if language == "sql":
                        result_dict = execute_sql_tool(tool_call, queries, folder, conn, temp_folder, data_list, transform_method)
                    else:
                        exec(
                            "import pandas as pd\n"
                            "pd.set_option('display.max_columns', None)\n"
                            "pd.set_option('display.width', None)\n"
                            "pd.set_option('display.max_colwidth', None)",
                            state,
                        )
                        result_dict = execute_py_tool(tool_call, queries, folder, state, temp_folder, csv_list, transform_method)

                    execution_res = result_dict.get("execution_results", {})
                    output = execution_res.get("outputs", "")
                    if result_dict.get("tool_name", None) == "execute_code":
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
                    res_result = json.dumps({
                        "outputs": f"[ERROR] Exception occurred during tool execution:\n{tb}"
                    })

                if tmp_res is not None:
                    result = tmp_res
                input_messages.append({
                    "name": tool_call.function.name,
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": str(res_result),
                })
        else:
            with open(log_path, "a") as f:
                f.write(f"========Processed Info======== \n")
                f.write("prompt: \n")
                f.write(prompt)
                f.write("response: \n")
                f.write(message.content)
                f.write("\n\n")
            break

        with open(log_path, "a") as f:
            f.write(f"========Processed Info======== \n")
            f.write(f"prompt: \n{prompt}")
            f.write(f"code: {code} \n")
            f.write(f"message: {message.content} \n")
            f.write(f"output: {output} \n")
            f.write(f"result: {result} \n\n")

    if language == "sql" and conn is not None:
        conn.close()
        if db_path and os.path.exists(db_path):
            os.remove(db_path)

    if result is not None:
        if isinstance(result, (int, float, str)):
            result = pd.DataFrame({"result": [result]})
        gt_dir = os.path.join(data_dir, f"cases_{transform_method}", "gt")
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

    if final_res:
        serializable_msgs = []
        for idx, msg in enumerate(input_messages):
            try:
                serializable = msg.to_dict() if hasattr(msg, "to_dict") else msg
                json.dumps(serializable)
                serializable_msgs.append(serializable)
            except Exception as e:
                print(f"[WARNING] Skipped message at index {idx}: {e}")

        print(f"Successfully serialized {len(serializable_msgs)} / {len(input_messages)} messages.")

        trace_dir = Path(output_dir) / f"{model}_{language}_{mode_suffix}" / transform_method
        trace_dir.mkdir(parents=True, exist_ok=True)

        with open(trace_dir / f"{folder}.json", "w") as f:
            json.dump(serializable_msgs, f, indent=2)

    with open(log_path, "a") as f:
        f.write("tool call seq:\n")
        f.write(str(tool_call_seq))
        f.write("\npred:\n")
        if isinstance(result, pd.DataFrame):
            f.write(result.head().to_string(index=False))
        else:
            f.write(str(result))
        f.write("\n\ngt:\n")
        try:
            gt_log = queries[folder]["gt"]
        except KeyError:
            gt_dir = os.path.join(data_dir, f"cases_{transform_method}", "gt")
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
    parser.add_argument("--id", required=True, help="Query ID")
    parser.add_argument("--model", required=True, help="Model to use (gpt-4o, gpt-4o-mini, o4-mini, o1-mini)")
    parser.add_argument("--language", choices=["python", "sql"], required=True)
    parser.add_argument("--runs", default=1, type=int, help="Number of attempts (stops early on success)")
    parser.add_argument("--transform", required=True, help="Transform type (transpose, pivot, unpivot, join)")
    parser.add_argument("--tool", action="store_true", help="Use tool-augmented prompts (default: baseline prompts)")
    parser.add_argument("--data-dir", default=".", help="Directory containing cases_* and training_cases_*.json")
    parser.add_argument("--output-dir", default="raw_traces", help="Root directory for saving successful traces")
    args = parser.parse_args()

    mode_str = "tool" if args.tool else "no-tool"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = f"logs/{args.transform}-{mode_str}/{args.model}_{args.language}_{args.id}_{timestamp}"
    os.makedirs(base_log_dir, exist_ok=True)

    for i in range(args.runs):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(base_log_dir, f"{i}.log")
        final_res, tool_call_seq = main(
            args.id, args.model, args.language, log_path,
            args.transform, args.tool, args.data_dir, args.output_dir,
        )
        if final_res:
            break
