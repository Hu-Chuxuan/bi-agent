from utils.llm_client import query_chat_endpoint
from utils.py_tools import execute_tool as execute_py_tool
from utils.sql_tools import execute_tool as execute_sql_tool

PROPRIETARY = {"gpt-4o", "o4-mini", "gpt-5.2"}
OPEN_SOURCE  = {"Llama-4-Maverick-17B-128E-Instruct-FP8"}
ALL_MODELS   = PROPRIETARY | OPEN_SOURCE


def is_open_source(model: str) -> bool:
    return model in OPEN_SOURCE


def call_model(model: str, messages, tools_spec):
    return query_chat_endpoint(model, messages, tools_spec)


def format_assistant_turn(response, text: str):
    """Message dict/object to append after the model replies."""
    return response.choices[0].message


def format_tool_turn(text: str, responses: list) -> dict:
    """Aggregated tool-response message (open-source style)."""
    return {"role": "tool", "content": str(responses)}


def format_tool_turn_proprietary(tc, res_result: str) -> dict:
    """Per-call tool-response message (proprietary style)."""
    return {
        "role": "tool",
        "tool_call_id": tc.id,
        "name": tc.function.name,
        "content": str(res_result),
    }
