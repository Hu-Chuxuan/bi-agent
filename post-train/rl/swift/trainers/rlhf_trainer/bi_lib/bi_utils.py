import json
import re
import os
import uuid
import shutil
import numpy as np
import pandas as pd

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
    """Parse tool call from SQL model output; handles ```json, ```sql, bare SELECT, and plain JSON."""
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
                code = tool_call_string.strip()
                return [ToolCall(function=ToolFunction(name='execute_code', arguments={"code": code}))]
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
                args_json = json.dumps(arguments)
            else:
                continue
            tool_calls.append(ToolCall(function=ToolFunction(name=func_name, arguments=args_json)))

        return tool_calls

    except Exception as e:
        print(f"[parse_tool_call_string] Error: {e}")
        return []


def parse_tool_call_string_python(tool_call_string: str) -> List[ToolCall]:
    """Parse tool call from Python model output; handles ```json, ```python, bare import, and plain JSON."""
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
                code = tool_call_string.strip()
                return [ToolCall(function=ToolFunction(name='execute_code', arguments={"code": code}))]
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
                args_json = json.dumps(arguments)
            else:
                continue
            tool_calls.append(ToolCall(function=ToolFunction(name=func_name, arguments=args_json)))

        return tool_calls

    except Exception as e:
        print(f"[parse_tool_call_string] Error: {e}")
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
    for df1 in gt_list:
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

        if all(used):
            return True

    return False


def create_temp_folder(folder):
    unique_id = uuid.uuid4().hex
    folder_name = f"{folder}_{unique_id}"
    src_path = os.path.join(os.environ.get("RFT_DATA_DIR", "rft_data"), folder)
    dst_path = f"./temp/{folder_name}/"

    os.makedirs(dst_path, exist_ok=True)
    if os.path.exists(src_path):
        for root, dirs, files in os.walk(src_path):
            rel_path = os.path.relpath(root, src_path)
            target_dir = os.path.join(dst_path, rel_path)
            os.makedirs(target_dir, exist_ok=True)
            for file in files:
                shutil.copy2(os.path.join(root, file), os.path.join(target_dir, file))

    return dst_path


def clean_tool_call_str(s: str) -> str:
    if "\\n" in s or "\n" not in s:
        return s
    return s.replace("\n", "\\n").replace("\r", "\\r")


def sanitize_name(name: str) -> str:
    name = name.replace("'", "").replace('"', '')
    name = re.sub(r'[^A-Za-z0-9]', '_', name)
    if re.match(r'^\d', name):
        name = '_' + name
    return name
