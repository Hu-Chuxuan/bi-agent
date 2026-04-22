from utils.llm_posttrained import query_chat_endpoint as _call
from utils.py_tools import execute_tool as execute_py_tool
from utils.sql_tools import execute_tool as execute_sql_tool

ALL_MODELS = set()  # model path is set via POSTTRAINED_MODEL_PATH env var


def is_open_source(model: str) -> bool:
    return True  # posttrained models always use text-based tool-call parsing


def call_model(model: str, messages, tools_spec):
    return _call(messages)


def format_assistant_turn(response, text: str) -> dict:
    return {"role": "assistant", "content": text}


def format_tool_turn(text: str, responses: list) -> dict:
    return {"role": "user", "content": f"The responses of {text} is {str(responses)}"}


# Proprietary-style per-call turn — unused for posttrained models
def format_tool_turn_proprietary(tc, res_result: str) -> dict:
    raise NotImplementedError("Posttrained models do not use proprietary tool-call format")
