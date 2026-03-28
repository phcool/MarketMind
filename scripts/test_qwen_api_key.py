"""Smoke test: Qwen via DashScope OpenAI-compatible API (streaming). Loads .env like the dashboard."""

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    path = _ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()

from openai import OpenAI

api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHBOARD_API_KEY")
if not api_key:
    raise SystemExit("Set DASHSCOPE_API_KEY or DASHBOARD_API_KEY in .env or environment.")

client = OpenAI(
    api_key=api_key,
    base_url=os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
)

model = os.getenv("DASHSCOPE_MODEL", "qwen3-max")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "请用一句话介绍你自己。"},
]

kwargs = dict(model=model, messages=messages, stream=True)
try:
    stream = client.chat.completions.create(**kwargs, stream_options={"include_usage": True})
except TypeError:
    stream = client.chat.completions.create(**kwargs)

print("AI: ", end="", flush=True)
parts: list[str] = []
for chunk in stream:
    if chunk.choices:
        content = chunk.choices[0].delta.content or ""
        print(content, end="", flush=True)
        parts.append(content)
    elif getattr(chunk, "usage", None):
        u = chunk.usage
        print("\n--- usage ---")
        print(f"prompt_tokens={getattr(u, 'prompt_tokens', None)} completion_tokens={getattr(u, 'completion_tokens', None)} total_tokens={getattr(u, 'total_tokens', None)}")

print("\n--- full ---\n", "".join(parts), sep="")
