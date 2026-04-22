import json
import io
import os
import traceback
import contextlib
import shutil
import pandas as pd

from .bi_utils import sanitize_name

def execute_tool(tool_call, queries, id, state, temp_folder, csv_list):

    tool_function_map = {
        "execute_code": execute_code,
        "retrieve_relevant_tables": retrieve_relevant_tables,
        "discover_join_relationships": discover_join_relationships,
        "transform_tables": transform_tables
    }

    tool_name = tool_call.function.name
    tool_args_json = tool_call.function.arguments
    tool_args = json.loads(tool_args_json)

    if tool_name not in tool_function_map:
        return {
            "tool_name": tool_name,
            "execution_results": {"outputs": f"[Error] Tool '{tool_name}' not found in function map."}
        }

    tool_fn = tool_function_map[tool_name]

    if tool_name == "retrieve_relevant_tables":
        execution_res = tool_fn(queries, id, temp_folder)
    elif tool_name == "transform_tables":
        execution_res = tool_fn(queries, id, temp_folder)
    elif tool_name == "discover_join_relationships":
        table_list = tool_args.get("table_list", csv_list)
        execution_res = tool_fn(id, temp_folder, table_list)
    else:
        execution_res = tool_fn(**tool_args, state=state)
    
    return {
        "tool_name": tool_name,
        "execution_results": execution_res
    }

def transform_tables(queries, id, temp_folder):
    transformations = queries[id].get("transformation", None)

    if transformations is None:
        return {
            "outputs": "No transformation is needed."
        }

    output_lines = []
    output_lines.append("The following table transformations were applied:\n")

    for transformation_name, table_list in transformations.items():
        output_lines.append(f"=== Transformation: {transformation_name} ===\n")
        for table_name in table_list:
            original_path = os.path.join(temp_folder, table_name)
            transformed_path = f"/datadrive/chuxuan/ms-swift/rft_data/transformation/{id}/{transformation_name}/{table_name}"

            try:
                df_before = pd.read_csv(original_path)
            except Exception:
                df_before = pd.DataFrame(["[Error loading original table]"])

            # Copy transformed table to destination
            shutil.copy2(transformed_path, original_path)

            try:
                df_after = pd.read_csv(transformed_path)
            except Exception:
                df_after = pd.DataFrame(["[Error loading transformed table]"])

            output_lines.append(f"Table: {table_name}")
            output_lines.append("- Before transformation:")
            output_lines.append(df_before.head(5).to_string(index=False))
            output_lines.append("- After transformation:")
            output_lines.append(df_after.head(5).to_string(index=False))
            output_lines.append("")  # Blank line for spacing

    return {
        "outputs": "\n".join(output_lines)
    }

def execute_code(code: str, state: dict = None):

    buffer = io.StringIO()
    safe_globals = {} if state is None else dict(state)  # Start from previous state

    try:
        code = code.replace('; ', '\n')
        code = code.encode().decode('unicode_escape')
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(code, safe_globals)
        result_value = safe_globals.get("result", None)
        return {
            "code": code,
            "result": result_value,
            "outputs": buffer.getvalue(),
            "state": safe_globals
        }
    except Exception:
        error_output = f"[ERROR] {traceback.format_exc()}"
        result_value = safe_globals.get("result", None)
        return {
            "code": code,
            "result": result_value,
            "outputs": error_output + "\n" + buffer.getvalue(),
            "state": safe_globals
        }

def retrieve_relevant_tables(queries, id, temp_folder):
    return {
        "outputs": [temp_folder + table for table in queries[id]["tables"]]
    }

def discover_join_relationships(id, temp_folder, table_list):
    relationships = parse_relationships(f"/datadrive/chuxuan/ms-swift/rft_data/{id}/relationships.tmdl")
    merge_statements = generate_join_statements(relationships, temp_folder, table_list)
    return {
        "outputs": merge_statements
    }

def parse_relationships(path):
    relationships = []
    current = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue  # Skip empty lines

            if stripped.startswith("relationship "):
                if current:
                    try:
                        if "joinOnDateBehavior" not in current:
                            relationships.append([current['fromColumn'], current['toColumn']])
                    except:
                        pass
                    current = {}
                current["id"] = stripped.split(" ")[1]
            elif ":" in stripped:
                key, value = stripped.split(":", 1)
                current[key.strip()] = value.strip()
            else:
                continue  # Skip lines without colon
        try:
            if current:
                relationships.append([current['fromColumn'], current['toColumn']])
        except:
            pass
    return relationships

def generate_join_statements(relationships, temp_folder, table_list):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    merge_statements = []
    # missing_files = set()
    for from_col, to_col in relationships:
        try:
            table1, col1 = from_col.split(".")
            table2, col2 = to_col.split(".")

            table1_csvname = table1.strip().strip("'").strip('"').replace("''", "'")
            table2_csvname = table2.strip().strip("'").strip('"').replace("''", "'")
            path1 = f"{temp_folder}{table1_csvname}.csv"
            path2 = f"{temp_folder}{table2_csvname}.csv"
            # print("DEBUG", path1)
            # print("DEBUG", table_list)
            if path1 not in table_list or path2 not in table_list:
                continue
            if not os.path.exists(path1):
                continue
            if not os.path.exists(path2):
                continue

            table1_df = pd.read_csv(path1)
            table2_df = pd.read_csv(path2)
            table1_df.columns = [sanitize_name(col) for col in table1_df.columns]
            table2_df.columns = [sanitize_name(col) for col in table2_df.columns]
            table1_df.to_csv(path1, index=False)
            table2_df.to_csv(path2, index=False)

            table1_varname = sanitize_name(table1)
            table2_varname = sanitize_name(table2)
            col1 = sanitize_name(col1)
            col2 = sanitize_name(col2)

            if col1 == col2:
                stmt = f"pd.merge({table1_varname}, {table2_varname}, on='{col1}')"
                merged_table = pd.merge(table1_df, table2_df, on=col1)
            else:
                stmt = f"pd.merge({table1_varname}, {table2_varname}, left_on='{col1}', right_on='{col2}')"
                merged_table = pd.merge(table1_df, table2_df,  left_on=col1, right_on=col2)
            merged_display = (
                f"\n# --- Table: {table1_varname} ---\n"
                f"{table1_varname} = pd.read_csv('{path1}')\n\n"
                f"{table1_df.head()}\n\n"
                f"# --- Table: {table2_varname} ---\n"
                f"{table2_varname} = pd.read_csv('{path2}')\n\n"
                f"{table2_df.head()}\n\n"
                f"# --- Merge Statement ---\n"
                f"{stmt}\n\n"
                f"# --- Resulting Merged Table (first 5 rows) ---\n"
                f"{merged_table.head()}\n"
            )
            merge_statements.append(merged_display)
        except:
            # Skip malformed entries
            continue
    # print(missing_files)
    return merge_statements