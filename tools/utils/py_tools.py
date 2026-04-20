import json
import io
import os
import re
import traceback
import contextlib
import difflib
import shutil
import glob
import csv
import pandas as pd
from collections import defaultdict, deque

from .file2var import sanitize_name
from .llm_client import query_chat_endpoint as llm_query

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(_UTILS_DIR), "data_management_tools")
_METADATA_DIR = os.path.join(_DATA_DIR, "metadata")
_JOIN_DIR = os.path.join(_DATA_DIR, "join")
_TRANSFORM_DIR = os.path.join(_DATA_DIR, "transformation")


def execute_tool(tool_call, queries, id, state, temp_folder, csv_list):
    tool_function_map = {
        "execute_code": execute_code,
        "retrieve_relevant_tables": retrieve_relevant_tables,
        "discover_join_relationships": discover_join_relationships,
        "transform_tables": transform_tables,
    }

    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)

    if tool_name not in tool_function_map:
        return {
            "tool_name": tool_name,
            "execution_results": {"outputs": f"[Error] Tool '{tool_name}' not found in function map."},
        }

    query = queries[id]
    tool_fn = tool_function_map[tool_name]
    if tool_name == "retrieve_relevant_tables":
        execution_res = tool_fn(query, temp_folder)
    elif tool_name == "transform_tables":
        execution_res = tool_fn(id, temp_folder)
    elif tool_name == "discover_join_relationships":
        table_list = tool_args.get("table_list", csv_list)
        execution_res = tool_fn(id, temp_folder, table_list)
    else:
        execution_res = tool_fn(**tool_args, state=state)

    return {"tool_name": tool_name, "execution_results": execution_res}


def transform_tables(id, temp_folder):
    base_path = os.path.join(_TRANSFORM_DIR, id)

    if not os.path.exists(base_path):
        return {"outputs": "No transformation is needed."}

    transformations = {}
    for transformation in os.listdir(base_path):
        trans_path = os.path.join(base_path, transformation)
        if os.path.isdir(trans_path):
            table_list = [f for f in os.listdir(trans_path) if f.endswith(".csv")]
            if table_list:
                transformations[transformation] = table_list

    if not transformations:
        return {"outputs": "No transformation is needed."}

    output_lines = ["The following table transformations were applied:\n"]
    for transformation_name, table_list in transformations.items():
        output_lines.append(f"=== Transformation: {transformation_name} ===\n")
        for table_name in table_list:
            original_path = os.path.join(temp_folder, table_name)
            transformed_path = os.path.join(_TRANSFORM_DIR, id, transformation_name, table_name)

            try:
                df_before = pd.read_csv(original_path)
            except Exception:
                df_before = pd.DataFrame(["[Error loading original table]"])

            shutil.copy2(transformed_path, original_path)

            try:
                df_after = pd.read_csv(transformed_path)
            except Exception:
                df_after = pd.DataFrame(["[Error loading transformed table]"])

            output_lines += [
                f"Table: {table_name}",
                "- Before transformation:",
                df_before.head(5).to_string(index=False),
                "- After transformation:",
                df_after.head(5).to_string(index=False),
                "",
            ]

    return {"outputs": "\n".join(output_lines)}


def execute_code(code: str, state: dict = None):
    buffer = io.StringIO()
    safe_globals = {} if state is None else dict(state)

    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(code, safe_globals)
        return {
            "code": code,
            "result": safe_globals.get("result", None),
            "outputs": buffer.getvalue(),
            "state": safe_globals,
        }
    except Exception:
        return {
            "code": code,
            "result": safe_globals.get("result", None),
            "outputs": f"[ERROR] {traceback.format_exc()}\n{buffer.getvalue()}",
            "state": safe_globals,
        }


def search_data_context(query, df_dict, model="gpt-4o"):
    table_context_str = "".join(
        f"### file_name:{name}\n\n{df.head().to_markdown(index=False)}\n\n\n\n"
        for name, df in df_dict.items()
    )
    prompt = f"""
    You are a data analyst. Given the following query, search for relevant data in the data context.
    Query: {query}
    Data context:
    {table_context_str}
    Think about what data files are relevant to the query, and return a list of relevant data files in JSON format: {{"files": LIST-OF-RELEVANT-FILE-NAMES}}
    **NOTE1:** When you are unsure about if some files contain duplicated information (e.g., the dates/calendars/geohraphic info), you should include them as well.
    **NOTE2:** When you return the list, MAKE SURE TO COPY THE CSV FILENAME AS IT IS (pay special attention to cases, blanks, etc).
    """
    input_messages = [{"role": "user", "content": prompt}]
    response = llm_query(model, input_messages)

    response_content = response.choices[0].message.content.strip()
    print(response_content)

    try:
        relevant_files = json.loads(response_content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", response_content, re.DOTALL)
        relevant_files = {}
        if match:
            try:
                relevant_files = json.loads(match.group(0))
            except Exception:
                pass

    relevant_file_list = []
    if isinstance(relevant_files, dict) and "files" in relevant_files:
        for file in relevant_files["files"]:
            if file in df_dict:
                relevant_file_list.append(file)
            else:
                closest = difflib.get_close_matches(file, df_dict.keys(), n=1, cutoff=0.0)
                if closest:
                    relevant_file_list.append(closest[0])
    else:
        print("Response does not contain 'files' key or is not a dictionary.")

    return relevant_file_list


def retrieve_relevant_tables(query, temp_folder):
    csv_list = [
        f for f in glob.glob(os.path.join(temp_folder, "*.csv"))
        if not os.path.basename(f).startswith("_")
    ]
    df_list = {}
    for _csv in csv_list:
        try:
            table_name = os.path.basename(_csv)
            df = pd.read_csv(_csv, low_memory=False, dtype=str, quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
            df.columns = [sanitize_name(col) for col in df.columns]
            df_list[table_name] = df
        except Exception:
            pass
    relevant_file_list = search_data_context(query, df_list)
    return {"outputs": [os.path.join(temp_folder, t) for t in relevant_file_list]}


def parse_relationships(id):
    column_df = pd.read_csv(os.path.join(_METADATA_DIR, id, "_columns.csv"))
    table_df = pd.read_csv(os.path.join(_METADATA_DIR, id, "_tables.csv"))
    relationship_df = pd.read_csv(os.path.join(_JOIN_DIR, f"{id}.csv"))
    return resolve_relationship_strings(relationship_df, column_df, table_df)


def resolve_relationship_strings(relationship_df, column_df, table_df):
    table_id_to_name = table_df.set_index("ID")["Name"].to_dict()

    def resolve_column_name(row):
        explicit = row["ExplicitName"]
        inferred = row["InferredName"]
        return explicit if pd.notna(explicit) and explicit != "" else inferred

    column_df["ResolvedName"] = column_df.apply(resolve_column_name, axis=1)
    col_id_to_name = column_df.set_index("ID")[["ResolvedName", "TableID"]].to_dict("index")

    result_pairs = []
    for _, row in relationship_df.iterrows():
        try:
            from_info = col_id_to_name[row["from_col"]]
            to_info = col_id_to_name[row["to_col"]]
            from_table = table_id_to_name.get(from_info["TableID"], f"UnknownTable_{from_info['TableID']}")
            to_table = table_id_to_name.get(to_info["TableID"], f"UnknownTable_{to_info['TableID']}")
            result_pairs.append([f"{from_table}.{from_info['ResolvedName']}", f"{to_table}.{to_info['ResolvedName']}"])
        except KeyError:
            continue

    return result_pairs


def boost_selection(table_list, id):
    relationships = parse_relationships(id)
    graph = defaultdict(set)
    for from_col, to_col in relationships:
        try:
            t1, _ = from_col.split(".")
            t2, _ = to_col.split(".")
            t1_csv = t1.strip().strip("'\"").replace("''", "'") + ".csv"
            t2_csv = t2.strip().strip("'\"").replace("''", "'") + ".csv"
            graph[t1_csv].add(t2_csv)
            graph[t2_csv].add(t1_csv)
        except Exception:
            continue

    def are_all_connected(graph, nodes):
        if not nodes:
            return True
        visited, q = {nodes[0]}, deque([nodes[0]])
        while q:
            for nb in graph[q.popleft()]:
                if nb in nodes and nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        return all(t in visited for t in nodes)

    if are_all_connected(graph, table_list):
        return []

    def bfs_path(start, goal):
        visited, q = set(), deque([(start, [start])])
        while q:
            node, path = q.popleft()
            if node == goal:
                return path
            visited.add(node)
            for nb in graph[node]:
                if nb not in visited:
                    q.append((nb, path + [nb]))
        return []

    boost_tables = set()
    for i in range(len(table_list)):
        for j in range(i + 1, len(table_list)):
            for t in bfs_path(table_list[i], table_list[j]):
                if t not in table_list:
                    boost_tables.add(t)

    return list(boost_tables)


def discover_join_relationships(id, temp_folder, table_list):
    tabname_list = [os.path.basename(_csv) for _csv in table_list]
    table_list = table_list + [os.path.join(temp_folder, t) for t in boost_selection(tabname_list, id)]
    relationships = parse_relationships(id)
    merge_statements = generate_join_statements(relationships, temp_folder, table_list)
    return {"outputs": {"merge statements": merge_statements, "full table list": table_list}}


def generate_join_statements(relationships, temp_folder, table_list):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    merge_statements = []
    for from_col, to_col in relationships:
        try:
            table1, col1 = from_col.split(".")
            table2, col2 = to_col.split(".")
            t1_csv = table1.strip().strip("'\"").replace("''", "'")
            t2_csv = table2.strip().strip("'\"").replace("''", "'")
            path1 = os.path.join(temp_folder, f"{t1_csv}.csv")
            path2 = os.path.join(temp_folder, f"{t2_csv}.csv")
            if path1 not in table_list or path2 not in table_list:
                continue
            if not os.path.exists(path1) or not os.path.exists(path2):
                continue

            df1 = pd.read_csv(path1)
            df2 = pd.read_csv(path2)
            df1.columns = [sanitize_name(c) for c in df1.columns]
            df2.columns = [sanitize_name(c) for c in df2.columns]
            v1, v2 = sanitize_name(table1), sanitize_name(table2)
            col1, col2 = sanitize_name(col1), sanitize_name(col2)

            if col1 == col2:
                stmt = f"pd.merge({v1}, {v2}, on='{col1}')"
                merged = pd.merge(df1, df2, on=col1)
            else:
                stmt = f"pd.merge({v1}, {v2}, left_on='{col1}', right_on='{col2}')"
                merged = pd.merge(df1, df2, left_on=col1, right_on=col2)

            merge_statements.append(
                f"\n# --- Table: {v1} ---\n{v1} = pd.read_csv({path1})\n\n{df1.head()}\n\n"
                f"# --- Table: {v2} ---\n{v2} = pd.read_csv({path2})\n\n{df2.head()}\n\n"
                f"# --- Merge Statement ---\n{stmt}\n\n"
                f"# --- Resulting Merged Table (first 5 rows) ---\n{merged.head()}\n"
            )
        except Exception:
            continue

    return merge_statements
