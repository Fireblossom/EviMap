from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def load_documents(path: Path, max_docs: int | None = None) -> list[dict]:
    docs: list[dict] = []
    for i, row in enumerate(read_jsonl(path)):
        doc_id = str(row.get("doc_id") or f"doc-{i:06d}")
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        docs.append({
            "doc_id": doc_id,
            "text": text,
            "metadata": row.get("metadata") or {},
        })
        if max_docs is not None and len(docs) >= max_docs:
            break
    if not docs:
        raise ValueError(f"no usable documents found in {path}")
    return docs


def slug(text: str, default: str = "item") -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or default

