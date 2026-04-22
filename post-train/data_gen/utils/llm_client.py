import os
import json
import time
import random
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

_GPT_MODELS = {"gpt-4o"}
_O_MODELS = {"o4-mini"}


def _load_clients(env_var: str) -> list:
    raw = os.environ.get(env_var, "")
    if not raw:
        raise EnvironmentError(
            f"{env_var} is not set. "
            f"Set it to a JSON array of {{\"endpoint\": \"...\", \"key\": \"...\"}} objects."
        )
    return [
        AzureOpenAI(api_version="2024-12-01-preview", azure_endpoint=c["endpoint"], api_key=c["key"])
        for c in json.loads(raw)
    ]


_gpt_clients: list | None = None
_o_clients: list | None = None


def _get_clients(model: str) -> list:
    global _gpt_clients, _o_clients
    if model in _GPT_MODELS:
        if _gpt_clients is None:
            _gpt_clients = _load_clients("AZURE_OPENAI_GPT_CONFIGS")
        return _gpt_clients
    if model in _O_MODELS:
        if _o_clients is None:
            _o_clients = _load_clients("AZURE_OPENAI_O_CONFIGS")
        return _o_clients
    raise ValueError(f"Unsupported model: {model}. Supported: {_GPT_MODELS | _O_MODELS}")


def query_chat_endpoint(model: str, input_messages, tools=None):
    clients = _get_clients(model)
    max_retries = 10
    result = None
    for attempt in range(1, max_retries + 1):
        try:
            client = random.choice(clients)
            kwargs = dict(model=model, messages=input_messages)
            if tools is not None:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            result = client.chat.completions.create(**kwargs)
            break
        except Exception as e:
            print(f"An error occurred: {e}. Retrying ({attempt}/{max_retries})...")
            time.sleep(10 * attempt)

    if result is None:
        print("Max retries reached. Skipping...")
        return None

    print(result.choices[0].message.content)
    return result
