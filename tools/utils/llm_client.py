"""
Unified LLM client. Define your own clients below before use.
"""

import time

# ---------------------------------------------------------------------------
# TODO: define your clients here
# ---------------------------------------------------------------------------

client = None    # e.g. AzureOpenAI(...)

# Models that use the Responses API instead of Chat Completions
_RESPONSES_API_MODELS = {"gpt-5.2"}

# ---------------------------------------------------------------------------
# Responses-API schema helper (required for gpt-5.2 strict mode)
# ---------------------------------------------------------------------------

def _enforce_no_additional_props(schema: dict) -> dict:
    if not isinstance(schema, dict):
        return schema
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
        for k, v in schema.get("properties", {}).items():
            schema["properties"][k] = _enforce_no_additional_props(v)
    if schema.get("type") == "array" and "items" in schema:
        schema["items"] = _enforce_no_additional_props(schema["items"])
    for key in ("oneOf", "anyOf", "allOf"):
        if key in schema and isinstance(schema[key], list):
            schema[key] = [_enforce_no_additional_props(s) for s in schema[key]]
    return schema


def _to_responses_tools(tools):
    if tools is None:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict) or "type" not in t:
            continue
        if t.get("type") == "function" and "name" in t:
            if isinstance(t.get("parameters"), dict):
                t["parameters"] = _enforce_no_additional_props(t["parameters"])
            out.append(t)
        elif t.get("type") == "function" and isinstance(t.get("function"), dict):
            f = t["function"]
            params = _enforce_no_additional_props(
                f.get("parameters", {"type": "object", "properties": {}})
            )
            out.append({
                "type": "function",
                "name": f.get("name"),
                "description": f.get("description", ""),
                "parameters": params,
                "strict": f.get("strict", True),
            })
        else:
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def query_chat_endpoint(model: str, input_messages, tools=None, reasoning_effort: str = "none"):
    max_retries = 5
    result = None

    for attempt in range(1, max_retries + 1):
        try:
            if model.startswith("Llama-"):
                kwargs = dict(model=model, messages=input_messages,
                              max_tokens=2048, temperature=0.8, top_p=0.1,
                              presence_penalty=0.0, frequency_penalty=0.0)
                if tools is not None:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                result = client.complete(**kwargs)

            elif model in _RESPONSES_API_MODELS:
                kwargs = dict(model=model, input=input_messages, reasoning={"effort": reasoning_effort})
                if tools is not None:
                    kwargs["tools"] = _to_responses_tools(tools)
                    kwargs["tool_choice"] = "auto"
                result = client.responses.create(**kwargs)

            else:
                kwargs = dict(model=model, messages=input_messages)
                if tools is not None:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                result = client.chat.completions.create(**kwargs)

            return result

        except Exception as e:
            print(f"[llmclient] attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(10 * attempt)

    print("[llmclient] Max retries reached.")
    return result
