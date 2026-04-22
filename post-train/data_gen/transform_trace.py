import json
import argparse
from pathlib import Path
from utils.prompt import TOOLS_PY, DEFAULT_SYSTEM_PROMPT, FUNCTION_STOPPER


def _json_dump(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def transform_trace(path_or_file) -> list[dict]:
    """Convert a ChatGPT-style trace to the format required by the downstream tool."""
    if isinstance(path_or_file, (str, Path)):
        with open(path_or_file, "r", encoding="utf-8") as f:
            raw_traces = json.load(f)
    else:
        raw_traces = json.load(path_or_file)

    transformed: list[dict] = []
    system_prompt = DEFAULT_SYSTEM_PROMPT + str(TOOLS_PY + FUNCTION_STOPPER)
    transformed.append({"role": "system", "content": system_prompt})

    i = 0
    while i < len(raw_traces):
        msg = raw_traces[i]
        role = msg.get("role")

        if role == "user":
            transformed.append(msg)
            i += 1
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                call_block = [
                    {
                        "function": call["function"]["name"],
                        "arguments": json.loads(call["function"]["arguments"] or "{}"),
                    }
                    for call in tool_calls
                ]
                formatted_calls = "```json\n" + _json_dump(call_block) + "\n```"
                transformed.append({"role": "assistant", "content": formatted_calls})

                responses = []
                j = i + 1
                while j < len(raw_traces) and raw_traces[j]["role"] == "tool":
                    responses.append(raw_traces[j]["content"])
                    j += 1

                transformed.append(
                    {
                        "role": "user",
                        "content": f"The responses of {formatted_calls} is {responses}",
                    }
                )
                i = j
                continue

            transformed.append(msg)
            i += 1
            continue

        i += 1

    return transformed


def transform_and_dump(
    path_to_trace: str,
    raw_dir: str = "raw_traces",
    output_dir: str = "transformed_traces",
) -> list[dict]:
    transformed = transform_trace(path_to_trace)
    src_path = Path(path_to_trace).resolve()

    try:
        raw_idx = src_path.parts.index(raw_dir)
    except ValueError:
        raise ValueError(f"'{raw_dir}' not found in path: {src_path}")

    relative_subpath = Path(*src_path.parts[raw_idx + 1:])
    new_root = Path(*src_path.parts[:raw_idx]) / output_dir
    out_path = new_root / relative_subpath
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(transformed)} messages to {out_path}")
    return transformed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", required=True, help="Path to raw trace JSON file")
    parser.add_argument("--raw-dir", default="raw_traces", help="Name of the raw traces root folder in the path")
    parser.add_argument("--output-dir", default="transformed_traces", help="Name of the output folder")
    args = parser.parse_args()

    transform_and_dump(args.filename, args.raw_dir, args.output_dir)
