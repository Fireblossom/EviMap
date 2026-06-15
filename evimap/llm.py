from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = os.getenv("EVIMAP_LLM_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
DEFAULT_BASE_URL = os.getenv("EVIMAP_LLM_BASE_URL", "http://127.0.0.1:18021/v1")
DEFAULT_TIMEOUT_S = float(os.getenv("EVIMAP_LLM_TIMEOUT_S", "180"))
DEFAULT_MAX_TOKENS = os.getenv("EVIMAP_LLM_MAX_TOKENS")

_LOCAL = threading.local()


def _api_key() -> str:
    if os.getenv("EVIMAP_LLM_API_KEY"):
        return os.environ["EVIMAP_LLM_API_KEY"]
    if os.getenv("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    key_path = Path.home() / ".config" / "openai_key"
    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()
    return "local"


def _extra_body() -> dict[str, Any] | None:
    raw = os.getenv("EVIMAP_LLM_EXTRA_BODY")
    if raw:
        return json.loads(raw)
    if "EVIMAP_LLM_ENABLE_THINKING" not in os.environ:
        return None
    enable = os.getenv("EVIMAP_LLM_ENABLE_THINKING", "false").lower()
    return {
        "chat_template_kwargs": {
            "enable_thinking": enable in {"1", "true", "yes", "on"},
        }
    }


def client():
    import httpx
    import openai

    cache_key = (_api_key(), DEFAULT_BASE_URL, DEFAULT_TIMEOUT_S)
    cached = getattr(_LOCAL, "client", None)
    cached_key = getattr(_LOCAL, "cache_key", None)
    if cached is not None and cached_key == cache_key:
        return cached

    timeout = httpx.Timeout(
        DEFAULT_TIMEOUT_S,
        connect=float(os.getenv("EVIMAP_LLM_CONNECT_TIMEOUT_S", "10")),
        write=DEFAULT_TIMEOUT_S,
        pool=float(os.getenv("EVIMAP_LLM_POOL_TIMEOUT_S", "10")),
    )
    kwargs: dict[str, Any] = {"api_key": _api_key(), "timeout": timeout}
    if DEFAULT_BASE_URL:
        kwargs["base_url"] = DEFAULT_BASE_URL
    _LOCAL.client = openai.OpenAI(**kwargs)
    _LOCAL.cache_key = cache_key
    return _LOCAL.client


def chat_json(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 2,
) -> dict:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model or DEFAULT_MODEL,
                "messages": messages,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }
            extra_body = _extra_body()
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            if DEFAULT_MAX_TOKENS:
                kwargs["max_tokens"] = int(DEFAULT_MAX_TOKENS)
            response = client().chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("LLM returned empty content")
            return json.loads(content)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            _LOCAL.client = None
            _LOCAL.cache_key = None
            if attempt < max_retries:
                time.sleep(1 + attempt)
    raise RuntimeError(f"LLM JSON call failed after retries: {last_error}") from last_error
