"""
Lazy-loading inference wrapper for HuggingFace causal LMs.

Usage:
    from utils.hf_model import query_chat_endpoint, MODEL_REGISTRY

    # model_id can be a short name from MODEL_REGISTRY or a direct HF/local path
    response = query_chat_endpoint(messages, model_id="infy-32b")
    response = query_chat_endpoint(messages, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct")
"""
import os
import torch

# Short name → HuggingFace model ID
MODEL_REGISTRY: dict[str, str] = {
    "infy-32b":  "infly/inf-rl-qwen-coder-32b-2746",
    "xiyan-32b": "XGenerationLab/XiYanSQL-QwenCoder-32B-2504",
    "kwai-autosql-14b": "Kwai-AutoSQL/Kwai-AutoSQL-14B",
    "kwai-autosql-32b": "Kwai-AutoSQL/Kwai-AutoSQL-32B",
    "llama3-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

_cache: dict[str, tuple] = {}  # model_id → (tokenizer, model)


def _resolve_model_id(model_id: str) -> str:
    return MODEL_REGISTRY.get(model_id, model_id)


def _load(model_id: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    _cache[model_id] = (tokenizer, model)


def query_chat_endpoint(input_messages, model_id: str, tools=None) -> str | None:
    resolved = _resolve_model_id(model_id)
    if resolved not in _cache:
        _load(resolved)
    tokenizer, model = _cache[resolved]

    text = tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    first_device = next(model.parameters()).device
    model_inputs = {k: v.to(first_device) for k, v in model_inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=2048)

    output_ids = generated_ids[0][len(model_inputs["input_ids"][0]):].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)
    return content
