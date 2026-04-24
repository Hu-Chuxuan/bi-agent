import json
import io
import os
import re
import traceback
import contextlib
import sqlite3
import sqlparse
import shutil
import pandas as pd
import concurrent.futures
from .bi_utils import sanitize_name

_RFT_DATA_DIR = os.environ.get("RFT_DATA_DIR", "rft_data")


def execute_tool(tool_call, queries, id, conn, temp_folder, data_list):

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
        execution_res = tool_fn(queries, id)
    elif tool_name == "transform_tables":
        execution_res = tool_fn(queries, id, temp_folder, conn)
    elif tool_name == "discover_join_relationships":
        data_list = tool_args.get("table_list", data_list)
        execution_res = tool_fn(id, temp_folder, data_list)
    else:
        execution_res = tool_fn(**tool_args, conn=conn)
        print(execution_res)
    
    return {
        "tool_name": tool_name,
        "execution_results": execution_res
    }

def transform_tables(queries, id, temp_folder, conn):
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
            transformed_path = os.path.join(_RFT_DATA_DIR, "transformation", id, transformation_name, table_name)
            table_name = os.path.basename(transformed_path)[:-4]
            table_name = sanitize_name(table_name)

            try:
                df_before = pd.read_csv(original_path)
                df_before.columns = [sanitize_name(col) for col in df_before.columns]
            except Exception:
                df_before = pd.DataFrame(["[Error loading original table]"])

            try:
                df_after = pd.read_csv(transformed_path)
                df_after.columns = [sanitize_name(col) for col in df_after.columns]
                df_after.to_sql(table_name, conn, if_exists='replace', index=False)
            except Exception:
                df_after = pd.DataFrame(["[Error loading transformed table]"])

            output_lines.append(f"Table: {table_name}")
            output_lines.append("- Before transformation (the first 5 rows):")
            output_lines.append(df_before.head(5).to_string(index=False))
            output_lines.append("- After transformation (the first 5 rows):")
            output_lines.append(df_after.head(5).to_string(index=False))
            output_lines.append("")  # Blank line for spacing

    return {
        "outputs": "\n".join(output_lines)
    }

def _execute_statements(code: str, conn: sqlite3.Connection):
    buffer = io.StringIO()
    result_value = None
    error_msg = ""

    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        cursor = conn.cursor()
        statements = sqlparse.split(code)

        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue
            try:
                cursor.execute(stmt)
                rows = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description] if cursor.description else []
                df = pd.DataFrame(rows, columns=col_names) if col_names else pd.DataFrame()
                print(f"\n[The first 5 rows for: {stmt}]\n{df.head()}\n")
                result_value = df
            except Exception as stmt_error:
                error_msg = f"\n[ERROR executing statement: {stmt}]\n{stmt_error}\n"
                break

        conn.commit()

    return {
        "code": code,
        "result": result_value,
        "outputs": buffer.getvalue() + error_msg
    }

def execute_code(code: str, conn: sqlite3.Connection, timeout: int = 300):
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_execute_statements, code, conn)
            return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return {
            "code": code,
            "result": None,
            "outputs": f"[ERROR] SQL execution exceeded {timeout} seconds and was terminated."
        }
    except Exception:
        tb = traceback.format_exc()
        return {
            "code": code,
            "result": None,
            "outputs": f"[ERROR] {tb}"
        }

#TODO: make it stored in intermediate value?
# def execute_code(code: str, conn: sqlite3.Connection):
#     buffer = io.StringIO()
#     result_value = None

#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.width', None)
#     pd.set_option('display.max_colwidth', None)

#     try:
#         with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
#             cursor = conn.cursor()
#             statements = sqlparse.split(code)
#             error_msg = ""

#             for stmt in statements:
#                 stmt = stmt.strip()
#                 if not stmt:
#                     continue
#                 try:
#                     cursor.execute(stmt)
#                     rows = cursor.fetchall()
#                     col_names = [desc[0] for desc in cursor.description]
#                     df = pd.DataFrame(rows, columns=col_names)
#                     print(f"\n[The first 5 rows for: {stmt}]\n{df.head()}\n")
#                     result_value = df
#                 except Exception as stmt_error:
#                     error_msg = f"\n[ERROR executing statement: {stmt}]\n{stmt_error}\n"
#                     break

#             conn.commit()

#             return {
#                 "code": code,
#                 "result": result_value,
#                 "outputs": buffer.getvalue() + error_msg
#             }

#     except Exception as e:
#         # capture full traceback including SQL context
#         tb = traceback.format_exc()
#         return {
#             "code": code,
#             "result": None,
#             "outputs": f"{buffer.getvalue()} \n\n [ERROR] {tb}"
#         }

def retrieve_relevant_tables(queries, id):
    data_list = []
    csv_list = queries[id]["tables"]
    for _csv in csv_list:
        try:
            table_name = os.path.basename(_csv)[:-4]
            table_name = sanitize_name(table_name)
            data_list.append(table_name)
        except:
            pass
    return {
        "outputs": data_list
    }

def discover_join_relationships(id, temp_folder, data_list):
    relationships = parse_relationships(os.path.join(_RFT_DATA_DIR, id, "relationships.tmdl"))
    merge_statements = generate_join_statements(relationships, temp_folder, data_list)
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

            table1_df = pd.read_csv(path1)
            table2_df = pd.read_csv(path2)
            table1_df.columns = [sanitize_name(col) for col in table1_df.columns]
            table2_df.columns = [sanitize_name(col) for col in table2_df.columns]

            table1_varname = sanitize_name(table1)
            table2_varname = sanitize_name(table2)

            if table1_varname not in table_list or table2_varname not in table_list:
                continue
            if not os.path.exists(path1):
                continue
            if not os.path.exists(path2):
                continue

            col1 = sanitize_name(col1)
            col2 = sanitize_name(col2)

            stmt = f'''SELECT {table1_varname}.*, {table2_varname}.* FROM {table1_varname} JOIN {table2_varname} ON {table1_varname}.{col1} = {table2_varname}.{col2};'''

            with sqlite3.connect(":memory:") as conn:
                table1_df.to_sql(table1_varname, conn, index=False, if_exists='replace')
                table2_df.to_sql(table2_varname, conn, index=False, if_exists='replace')
                merged_table = pd.read_sql_query(stmt, conn)

            # merged_display = f"To join {table1_varname} and {table2_varname}, use: {stmt}"
            merged_display = (
                f"\n# --- Table: {table1_varname} ---\n"
                f"SELECT * from {table1_varname} LIMIT 5;\n\n"
                f"{table1_df.head()}\n\n"
                f"# --- Table: {table2_varname} ---\n"
                f"SELECT * from {table2_varname} LIMIT 5;\n\n"
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
    # return "\n".join(merge_statements)
    return merge_statements