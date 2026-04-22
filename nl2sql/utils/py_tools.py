import json
import io
import traceback
import contextlib


def execute_tool(tool_call, queries, id, state, temp_folder, csv_list):
    tool_name = tool_call.function.name
    tool_args_json = tool_call.function.arguments
    tool_args = json.loads(tool_args_json)

    if tool_name != "execute_code":
        return {
            "tool_name": tool_name,
            "execution_results": {"outputs": f"[Error] Tool '{tool_name}' not found in function map."}
        }

    return {
        "tool_name": tool_name,
        "execution_results": execute_code(**tool_args, state=state)
    }


def execute_code(code: str, state: dict = None):
    buffer = io.StringIO()
    safe_globals = {} if state is None else dict(state)

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
