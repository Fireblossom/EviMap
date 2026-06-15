from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_BACKEND = os.getenv("EVIMAP_EMBEDDING_BACKEND", "local")
DEFAULT_MODEL = os.getenv(
    "EVIMAP_EMBEDDING_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2",
)
DEFAULT_DEVICE = os.getenv("EVIMAP_EMBEDDING_DEVICE", "auto")
DEFAULT_BASE_URL = os.getenv("EVIMAP_EMBEDDING_BASE_URL", "")


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return matrix / norms


def _api_key() -> str:
    if os.getenv("EVIMAP_EMBEDDING_API_KEY"):
        return os.environ["EVIMAP_EMBEDDING_API_KEY"]
    if os.getenv("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    key_path = Path.home() / ".config" / "openai_key"
    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()
    return "local"


def _resolve_device(device: str) -> str:
    device = (device or "auto").lower()
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _embed_local(texts: list[str], model_name: str, device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    resolved = _resolve_device(device)
    print(f"[embeddings] local model={model_name} device={resolved}")
    model = SentenceTransformer(model_name, device=resolved)
    return model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)


def _embed_openai(
    texts: list[str],
    model_name: str,
    base_url: str,
    batch_size: int,
) -> np.ndarray:
    import openai

    kwargs = {"api_key": _api_key()}
    if base_url:
        kwargs["base_url"] = base_url
    client = openai.OpenAI(**kwargs)
    rows: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=model_name, input=batch)
        rows.extend(item.embedding for item in response.data)
        print(f"[embeddings] embedded {min(start + len(batch), len(texts))}/{len(texts)}")
    return np.asarray(rows, dtype=np.float32)


def embed_texts(
    texts: Iterable[str],
    *,
    backend: str = DEFAULT_BACKEND,
    model_name: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    batch_size: int = 128,
    device: str = DEFAULT_DEVICE,
) -> np.ndarray:
    values = list(texts)
    if not values:
        return np.zeros((0, 0), dtype=np.float32)
    backend = (backend or "local").lower()
    if backend == "local":
        return _embed_local(values, model_name, device)
    if backend in {"openai", "remote"}:
        return _embed_openai(values, model_name, base_url, batch_size)
    raise ValueError(f"unknown embedding backend: {backend}")

