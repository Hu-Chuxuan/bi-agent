import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is not None:
        return
    model_path = os.environ.get("POSTTRAINED_MODEL_PATH")
    if not model_path:
        raise RuntimeError("POSTTRAINED_MODEL_PATH environment variable is not set")
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


def query_chat_endpoint(messages, tools=None):
    _load()
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = _tokenizer([text], return_tensors="pt").to(_model.device)
    generated_ids = _model.generate(**model_inputs, max_new_tokens=32768)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)  # </think>
    except ValueError:
        index = 0

    thinking = _tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content  = _tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    print("thinking:", thinking)
    print("content:", content)
    return content
