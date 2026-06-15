from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def _is_word_char(char: str) -> bool:
    return char.isascii() and char.isalnum()


def find_all_occurrences(text: str, phrase: str) -> list[tuple[int, int]]:
    """Case-insensitive substring search with ASCII word-boundary guards."""
    if not text or not phrase:
        return []
    text_low = text.lower()
    phrase_low = phrase.lower()
    check_left = _is_word_char(phrase[0])
    check_right = _is_word_char(phrase[-1])
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text_low.find(phrase_low, cursor)
        if start < 0:
            break
        end = start + len(phrase)
        left_ok = not check_left or start == 0 or not _is_word_char(text[start - 1])
        right_ok = not check_right or end >= len(text) or not _is_word_char(text[end])
        if left_ok and right_ok:
            spans.append((start, end))
            cursor = end
        else:
            cursor = start + 1
    return spans


def _list_text(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip() and item.strip() not in out:
                out.append(item.strip())
        return out
    return []


def build_phrase_index(
    documents: Iterable[dict],
    extraction_rows: Iterable[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    docs = {d["doc_id"]: d for d in documents}
    entry_by_norm: dict[str, dict] = {}
    occurrences: list[dict] = []
    unmatched: list[dict] = []
    role_counts: dict[str, Counter] = defaultdict(Counter)
    axis_pool: dict[str, set[str]] = defaultdict(set)
    doc_sets: dict[str, set[str]] = defaultdict(set)

    def entry_for(text: str) -> dict:
        norm = normalize_text(text)
        if norm not in entry_by_norm:
            entry_id = f"phrase-{len(entry_by_norm):06d}"
            entry_by_norm[norm] = {
                "phrase_entry_id": entry_id,
                "text": text.strip(),
                "text_normalized": norm,
                "support_doc_count": 0,
                "occurrence_count": 0,
                "role_hint_dist": {},
                "axis_hints_pool": [],
            }
        return entry_by_norm[norm]

    for row in extraction_rows:
        doc_id = str(row.get("doc_id") or "")
        doc = docs.get(doc_id)
        if not doc:
            continue
        text = doc["text"]
        seen_in_doc: set[str] = set()
        for phrase in row.get("phrases") or []:
            if not isinstance(phrase, dict):
                continue
            phrase_text = str(phrase.get("text") or "").strip()
            if not phrase_text:
                continue
            entry = entry_for(phrase_text)
            entry_id = entry["phrase_entry_id"]
            norm = entry["text_normalized"]
            role = str(phrase.get("role_hint") or "unclear").strip() or "unclear"
            axes = _list_text(phrase.get("axis_hints"))
            role_counts[entry_id][role] += 1
            axis_pool[entry_id].update(axes)
            if norm not in seen_in_doc:
                doc_sets[entry_id].add(doc_id)
                seen_in_doc.add(norm)
            spans = find_all_occurrences(text, phrase_text)
            if not spans:
                unmatched.append({
                    "doc_id": doc_id,
                    "phrase_entry_id": entry_id,
                    "text": phrase_text,
                    "reason": "substring_not_found",
                    "role_hint": role,
                    "axis_hints": axes,
                })
                continue
            for start, end in spans:
                occurrences.append({
                    "occurrence_id": f"occ-{len(occurrences):08d}",
                    "doc_id": doc_id,
                    "phrase_entry_id": entry_id,
                    "text": phrase_text,
                    "matched_text": text[start:end],
                    "start": start,
                    "end": end,
                    "role_hint": role,
                    "axis_hints": axes,
                    "context_note": str(phrase.get("context_note") or ""),
                })

    occ_counts = Counter(o["phrase_entry_id"] for o in occurrences)
    n_docs = len(docs)
    entries = list(entry_by_norm.values())
    for entry in entries:
        entry_id = entry["phrase_entry_id"]
        support = len(doc_sets[entry_id])
        entry["support_doc_count"] = support
        entry["occurrence_count"] = int(occ_counts[entry_id])
        entry["idf"] = math.log((1 + n_docs) / (1 + support)) + 1.0
        entry["role_hint_dist"] = dict(role_counts[entry_id])
        entry["axis_hints_pool"] = sorted(axis_pool[entry_id])

    entries.sort(key=lambda r: (-r["support_doc_count"], -r["occurrence_count"], r["text_normalized"]))
    for i, entry in enumerate(entries):
        entry["embedding_row"] = i
    return entries, occurrences, unmatched

