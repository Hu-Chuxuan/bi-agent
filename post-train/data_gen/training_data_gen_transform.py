# Script 1: pivot/unpivot candidate selection
# Script 1.5: transpose

# Script 2: question generation
#   randomly select connected components
#   give relationships
#   hardcode on transformation (check if the tables appear)
#   max 5 questions per folder; min the number of available components (discard the tables that are already selected and asked questions on)
#   constraints: keyword combinations/examples of our handlabeled queries

# Script 3: transform gpt-4o/o4-mini traces to llama style
from collections import defaultdict, deque
from pathlib import Path
import pandas as pd
import os
import re
import json
import shutil
import argparse
import multiprocessing as mp
import queue as queue_mod
import traceback

from utils.llm_client import query_chat_endpoint
from utils.file2var import sanitize_name

from utils.s1_reverse_engineer_transpose_for_training import re_transpose
from utils.s2_reverse_engineer_unpivot_for_training import re_unpivot
from utils.s3_reverse_engineer_pivot_for_training import re_pivot

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

def generate_max_components(id):
    # Step 1: Parse join relationships into a graph
    relationships = parse_relationships(f"./original/{id}/relationships.tmdl")
    graph = defaultdict(set)

    for from_col, to_col in relationships:
        try:
            table1 = from_col.split(".")[0]
            table2 = to_col.split(".")[0]
            table1_csvname = table1.strip().strip("'").strip('"').replace("''", "'")
            table2_csvname = table2.strip().strip("'").strip('"').replace("''", "'")
            graph[table1_csvname].add(table2_csvname)
            graph[table2_csvname].add(table1_csvname)
        except:
            continue
    
    print(graph)

    # Step 2: Find all connected components
    visited = set()
    components = []

    for node in graph:
        if node not in visited:
            q = deque([node])
            component = []
            visited.add(node)
            while q:
                curr = q.popleft()
                component.append(curr)
                for neighbor in graph[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            components.append(component)
    return components

def join_table(id, table_list, transform_method):
    relationships = parse_relationships(f"./cases_{transform_method}/{id}/relationships.tmdl")
    return join_from_relationships(relationships, id, table_list, transform_method)

def clean(s):
    return s.strip().strip("'").strip('"').replace("''", "'")

def join_from_relationships(relationships, id, table_list, transform_method):

    # Build adjacency graph from relationships
    graph = defaultdict(list)  # table -> list of (neighbor, this_col, neighbor_col)
    for from_col, to_col in relationships:
        try:
            t1, c1 = from_col.split(".", 1)
            t2, c2 = to_col.split(".", 1)
            t1, c1, t2, c2 = clean(t1), clean(c1), clean(t2), clean(c2)
            if t1 in table_list and t2 in table_list:
                graph[t1].append((t2, c1, c2))
                graph[t2].append((t1, c2, c1))
        except Exception:
            continue

    if not graph:
        raise ValueError("No usable relationships among the provided tables.")

    # Load each table and prefix columns with table name
    dfs = {}
    for t in table_list:
        path = f"./cases_{transform_method}/{id}/{t}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing table file: {path}")
        df = pd.read_csv(path)
        df.columns = [f"{t}.{col}" for col in df.columns]
        dfs[t] = df

    # Start merging from the first table in the list
    merged_df = dfs[table_list[0]].copy()
    merged_tables = {table_list[0]}

    # Iteratively attach remaining tables using relationships
    made_progress = True
    while merged_tables != set(table_list) and made_progress:
        made_progress = False
        for current in list(merged_tables):
            for neighbor, current_col, neighbor_col in graph[current]:
                if neighbor in merged_tables:
                    continue
                left_key = f"{current}.{current_col}"
                right_key = f"{neighbor}.{neighbor_col}"
                if left_key not in merged_df.columns or right_key not in dfs[neighbor].columns:
                    continue
                merged_df = pd.merge(
                    merged_df,
                    dfs[neighbor],
                    left_on=left_key,
                    right_on=right_key
                )
                merged_tables.add(neighbor)
                made_progress = True
                break
            if made_progress:
                break

    if merged_tables != set(table_list):
        raise RuntimeError("Could not merge all tables; check relationships/keys.")

    return merged_df

TIMEOUT = 30  # seconds

def _exec_code(code: str, state: dict, out_q: mp.Queue) -> None:
    try:
        safe_globals = {} if state is None else dict(state)
        exec(code, safe_globals)
        out_q.put(safe_globals.get("result", None))
    except Exception:
        out_q.put(None)

def execute_code(code: str, state: dict | None = None, timeout: int = TIMEOUT):
    out_q = mp.Queue()
    proc = mp.Process(target=_exec_code, args=(code, state or {}, out_q))
    proc.start()

    try:
        result = out_q.get(timeout=timeout)
    except queue_mod.Empty:
        proc.terminate()
        result = None

    proc.join()
    return result

def load_case_from_response(text: str):
    """
    Parse an LLM response that contains a ```json ... ``` fenced block whose
    "code" field is a multi‑line string. Returns a Python dict or None.
    """
    # 1) Strip ```json / ``` fences if present
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()

    # 2) Locate the raw code block (everything between the first quote after "code":)
    code_match = re.search(r'"code"\s*:\s*"([\s\S]*?)"', cleaned)
    if code_match:
        raw_code = code_match.group(1)

        # 3) Escape backslashes, quotes, and newlines so the string becomes valid JSON
        esc_code = (
            raw_code
            .replace("\\", "\\\\")    # backslash first
            .replace('"', r'\"')       # double quotes
            .replace("\n", r"\n")      # newlines
        )

        # 4) Splice the escaped code back into the JSON text
        cleaned = (
            cleaned[:code_match.start(1)] +
            esc_code +
            cleaned[code_match.end(1):]
        )

    # 5) Finally try to load the repaired JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

def infer_direction(tables, A_name, colA, B_name, colB):
    """
    Infer direction between two tables based on key uniqueness and inclusion.

    Args:
      tables: dict[str, pd.DataFrame], e.g. {"Orders": df1, "Customers": df2}
      A_name: str, name of first table (must be in tables)
      colA:   str, join key column in A_name
      B_name: str, name of second table (must be in tables)
      colB:   str, join key column in B_name

    Returns:
      dict with keys:
        'cardinality': '1:1' | '1:N' | 'N:1' | 'N:N'
        'from': table name (str)
        'to':   table name (str)
        'reasons': list[str]
    """
    dfA = tables[A_name]
    dfB = tables[B_name]

    reasons = []
    nA, nB = len(dfA), len(dfB)
    uA = dfA[colA].nunique(dropna=True)
    uB = dfB[colB].nunique(dropna=True)
    is_unique_A = (uA == nA)
    is_unique_B = (uB == nB)
    nullA = dfA[colA].isna().mean()
    nullB = dfB[colB].isna().mean()

    setA = set(dfA[colA].dropna().unique())
    setB = set(dfB[colB].dropna().unique())

    subset_A_in_B = setA.issubset(setB)
    subset_B_in_A = setB.issubset(setA)

    if is_unique_A and is_unique_B:
        cardinality = "1:1"
        if subset_A_in_B and not subset_B_in_A:
            frm, to = A_name, B_name
        elif subset_B_in_A and not subset_A_in_B:
            frm, to = B_name, A_name
        else:
            # tie-break: higher null rate is more likely the referencing side
            frm, to = (A_name, B_name) if nullA >= nullB else (B_name, A_name)
        reasons += ["both unique"]
    elif not is_unique_A and is_unique_B:
        cardinality = "N:1"
        frm, to = A_name, B_name
        reasons += ["A not unique", "B unique"]
    elif is_unique_A and not is_unique_B:
        cardinality = "1:N"
        frm, to = B_name, A_name   # from many to one
        reasons += ["B not unique", "A unique"]
    else:
        cardinality = "N:N"
        if subset_A_in_B and not subset_B_in_A:
            frm, to = A_name, B_name
            reasons += ["A subset of B"]
        elif subset_B_in_A and not subset_A_in_B:
            frm, to = B_name, A_name
            reasons += ["B subset of A"]
        else:
            frm, to = (A_name, B_name) if nullA >= nullB else (B_name, A_name)
            reasons += ["no clear subset; chose direction by higher nulls referencing lower nulls"]

    return {"cardinality": cardinality, "from": frm, "to": to, "reasons": reasons}


def generate_path_components(id):
    """
    Build a directed graph (from_table -> to_table) and extract all maximal
    simple paths that start at fact tables. A fact table is any node with
    global in-degree 0 (never appears as a 'to').

    A path is extended while nodes remain in a linear chain:
      current out-degree == 1 and next in-degree == 1.
    If a node has multiple outgoing edges, we start a separate path for each edge,
    but stop extension at the branch point to keep the component a path.

    Returns:
      List[dict] with keys:
        - 'fact_table': str
        - 'path': List[str]  ordered from start to end
    """
    relationships = parse_relationships(f"./original/{id}/relationships.tmdl")

    # Build directed graph
    graph = defaultdict(set)
    indeg = defaultdict(int)
    all_nodes = set()

    for from_col, to_col in relationships:
        try:
            t1, c1 = from_col.split(".", 1)
            t2, c2 = to_col.split(".", 1)
            t1, c1, t2, c2 = clean(t1), clean(c1), clean(t2), clean(c2)

            path1 = f"./original/{id}/{t1}.csv"
            path2 = f"./original/{id}/{t2}.csv"
            df1 = pd.read_csv(path1)
            df2 = pd.read_csv(path2)

            tables = {t1: df1, t2: df2}
            from_to_relationships = infer_direction(tables, t1, c1, t2, c2)
            print(from_to_relationships)

            if t2 not in graph[t1]:
                graph[from_to_relationships["from"]].add(from_to_relationships["to"])
                indeg[from_to_relationships["to"]] += 1
            all_nodes.update([from_to_relationships["from"], from_to_relationships["to"]])
        except Exception:
            continue

    # Ensure every node is present in structures
    for n in list(all_nodes):
        graph.setdefault(n, set())
        indeg.setdefault(n, 0)

    fact_tables = [n for n in all_nodes if indeg[n] == 0]

    components = []
    seen_sets = set()  # to dedupe identical reachable sets

    def reachable_from(src):
        """Return all nodes reachable from src following edge direction."""
        seen = set()
        stack = [src]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            for v in graph[u]:
                if v not in seen:
                    stack.append(v)
        return seen

    for f in sorted(fact_tables):
        reach = reachable_from(f)  # includes the fact table itself
        key = tuple(sorted(reach))
        if key in seen_sets:
            continue
        seen_sets.add(key)
        components.append({
            "fact_table": f,
            "nodes": sorted(reach)  # all tables reachable from f
        })

    return components
    
def generate_question(table, model):
    prompt = f"""
    Given the following table with head {table.head()},
    your task is to generate a natural language query that makes use of multiple (at least 2) tables, 
    as well as the pandas code to answer the natural language query.
    In your generated code, you should strictly assign your results to a `result` variable, and `result` can only be pd.DataFrame.
    The table is loaded as a df in a variable `data` for you already. Both your natural language query and your generated code should NOT involve generating new table(s).
    Your output should strictly be in json format:
    ```json
    {{
        "query":
        "code":
        "table": LIST-OF-TABLES, subset of provided tables
    }}
    ```
    """

    input_messages = [{"role": "user", "content": prompt}]
    print("DEBUG PROMPT*****", prompt)
    response = query_chat_endpoint(model, input_messages)
    
    # parse the response
    response_content = response.choices[0].message.content.strip()
    print(response_content)

    case = load_case_from_response(response_content)
    state = {"data": table}
    exec(
        "import pandas as pd\n"
        "pd.set_option('display.max_columns', None)\n"
        "pd.set_option('display.width', None)\n"
        "pd.set_option('display.max_colwidth', None)",
        state
    )

    try:
        code = case["code"].encode().decode('unicode_escape')
        return case["query"], case["table"], code, execute_code(code, state)
    except:
        return None, [], None, None

def _normalize_tables_for_case(tables):
    # Reduce to base table names so join_table(case_id, tables) loads from that case folder
    # (join_table should append .csv itself; if not, adapt accordingly)
    return [Path(t).stem for t in tables]

def generate_from_folder(id, model, transform_method):
    # Path for result JSON
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    res_path = f"./training_cases_{transform_method}.json"

    # Step 0: Load existing JSON if available
    if os.path.exists(res_path):
        with open(res_path, "r", encoding="utf-8") as f:
            try:
                res = json.load(f)
            except json.JSONDecodeError:
                res = {}
    else:
        res = {}

    # Ensure ground truth directory exists
    gt_dir = f"./cases_{transform_method}/gt"
    os.makedirs(gt_dir, exist_ok=True)

    # Step 1: Select tables
    # max_components = generate_max_components(id)
    max_components = generate_path_components(id)
    print(max_components)
    # return
    for i, comp in enumerate(max_components):  # comp: {'fact_table': str, 'path': List[str]}
        original_path_tables = comp["nodes"]
        fact_table_name = comp["fact_table"]

        if len(original_path_tables) < 2:
            continue

        src_dir = f"./original/{id}"
        tmp_id = f"{id}__tmp_{i}"
        tmp_dir = f"./cases_{transform_method}/{tmp_id}"
        dst_dir = f"./cases_{transform_method}/{id}-{i}"

        # fresh tmp workspace
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        shutil.copytree(src_dir, tmp_dir)

        # sample the fact table INSIDE tmp_dir so everything uses the sampled data
        fact_table_stem = Path(fact_table_name).stem
        fact_csv_in_tmp = None
        for p in Path(tmp_dir).glob("*.csv"):
            if p.stem == fact_table_stem:
                fact_csv_in_tmp = p
                break

        if fact_csv_in_tmp is not None:
            try:
                df_fact = pd.read_csv(fact_csv_in_tmp)
                n = min(100, len(df_fact))
                df_sample = df_fact.sample(n=n, random_state=42) if n > 0 else df_fact.head(0)
                df_sample.to_csv(fact_csv_in_tmp, index=False)
                print(f"Sampled {n} rows for fact table '{fact_table_stem}' in {fact_csv_in_tmp}")
            except Exception as e:
                print(f"Warning: could not sample fact table '{fact_table_stem}': {e}")
        else:
            print(f"Warning: fact table CSV for '{fact_table_stem}' not found in {tmp_dir}")

        # IMPORTANT: rewrite path_tables to bind them to the sampled case id
        # Use base names so join_table(tmp_id, ...) will load from ./cases/{tmp_id}/
        path_tables = _normalize_tables_for_case(original_path_tables)

        # Join using the sampled case
        joined_table = join_table(tmp_id, path_tables, transform_method)
        transform_candidates = [comp['fact_table']] + comp['nodes']
        transformed_table = None
        for transform_candidate in transform_candidates:
            transform_candidate_path = f"./cases_{transform_method}/{tmp_id}/{transform_candidate}.csv"
            df_candidate = pd.read_csv(transform_candidate_path)
            if transform_method == "transpose":
                transformed_table = re_transpose(df_candidate)
            elif transform_method == "unpivot":
                transformed_table = re_unpivot(df_candidate)
            elif transform_method == "pivot":
                transformed_table = re_pivot(df_candidate)
            else:
                raise RuntimeError("Invalid transformation!")
            if transformed_table is not None:
                break
        if transformed_table is None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        table = []
        fact_table = f"{fact_table_stem}.csv"
        max_retries = 3
        gt = None
        query = None
        code = None
        while (len(table) < 2 or f"{transform_candidate}.csv" not in table or fact_table not in table or gt is None or not isinstance(gt, pd.DataFrame) or gt.empty or gt.dropna(how="all").empty) and max_retries > 0:
            query, table, code, gt = generate_question(joined_table, model)
            table = [f"{Path(t).stem}.csv" if not Path(t).name.endswith(".csv") else Path(t).name for t in table]
            max_retries -= 1
            print("MAX_RETRIES", max_retries)

        print("Query:", query)
        print("Tables:", table)
        print("GT:", gt)

        if gt is None:
            # discard tmp if no usable result
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        # promote tmp_dir (with sampled fact table) to final dst_dir
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.move(tmp_dir, dst_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        entry_key = f"{id}-{i}"
        # normalize table names to .csv for storage
        table = [f"{Path(t).stem}.csv" if not Path(t).name.endswith(".csv") else Path(t).name for t in table]
        res[entry_key] = {
            "query": query,
            "tables": table,
            "transformation": {transform_method: [f"{transform_candidate}.csv"]},
            "fact_table": f"{fact_table_stem}.csv",
            "path_tables": [f"{t}.csv" for t in path_tables]  # these are the sampled-case tables
        }

        transform_dir = f"./cases_{transform_method}/transformation/{entry_key}/{transform_method}"
        os.makedirs(transform_dir, exist_ok=True)

        # move the original file into the transform_dir
        src_file = f"{dst_dir}/{transform_candidate}.csv"
        dst_file = f"{transform_dir}/{transform_candidate}.csv"
        shutil.move(src_file, dst_file)

        # write the new transformed file back into dst_dir
        transformed_table.to_csv(f"{dst_dir}/{transform_candidate}.csv", index=False)

        # Save ground truth
        if isinstance(gt, pd.DataFrame):
            gt_path = f"{gt_dir}/{entry_key}.csv"
            gt.to_csv(gt_path, index=False)
        else:
            res[entry_key]["gt"] = gt

        # Save code if provided
        if isinstance(code, str):
            code_path = os.path.join(gt_dir, f"{entry_key}.py")
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)

    # Step 2: Write res back to JSON
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="Query ID")
    parser.add_argument("--model", required=True, help="Model to use (e.g., deepseek)")
    parser.add_argument("--transform", required=True)

    args = parser.parse_args()
    generate_from_folder(args.id, args.model, args.transform)


