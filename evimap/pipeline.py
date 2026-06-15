from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__
from .embeddings import DEFAULT_BACKEND, DEFAULT_BASE_URL, DEFAULT_DEVICE, DEFAULT_MODEL as DEFAULT_EMBED_MODEL
from .embeddings import embed_texts, normalize_rows
from .io import ensure_dir, load_documents, write_json, write_jsonl
from .llm import DEFAULT_MODEL as DEFAULT_LLM_MODEL, chat_json
from .prompts import (
    EXTRACTION_SYSTEM,
    FINE_GROUP_SYSTEM,
    MID_GROUP_SYSTEM,
    NAME_ASPECT_SYSTEM,
    NAME_GROUP_SYSTEM,
    NAME_TOPIC_SYSTEM,
    PROFILE_SYSTEM,
    TOP_GROUP_SYSTEM,
)
from .spans import build_phrase_index


@dataclass
class PipelineConfig:
    input: str
    output: str
    model: str = DEFAULT_LLM_MODEL
    max_docs: int | None = None
    profile_sample_size: int = 8
    doc_truncate_chars: int = 12000
    workers: int = 4
    seed: int = 13
    rounds: int = 3
    chunk_size: int = 20
    min_coassoc: float = 0.67
    min_meet: int = 1
    phrase_domains: int = 8
    topic_domains: int = 4
    top_k: int = 12
    embedding_backend: str = DEFAULT_BACKEND
    embedding_model: str = DEFAULT_EMBED_MODEL
    embedding_base_url: str = DEFAULT_BASE_URL
    embedding_batch_size: int = 128
    embedding_device: str = DEFAULT_DEVICE
    max_name_members: int = 18


def discover_profile(docs: list[dict], cfg: PipelineConfig) -> dict:
    sample = docs[: cfg.profile_sample_size]
    user = "\n\n".join(
        f"--- Document {i + 1}: {d['doc_id']} ---\n{d['text'][:1600]}"
        for i, d in enumerate(sample)
    )
    return chat_json(PROFILE_SYSTEM, user, model=cfg.model, temperature=0.0)


def extract_document(doc: dict, profile: dict, cfg: PipelineConfig) -> dict:
    profile_text = json.dumps(profile, ensure_ascii=False, indent=2)
    user = (
        "Domain profile, used only as extraction guidance:\n"
        f"{profile_text}\n\n"
        f"Document id: {doc['doc_id']}\n"
        f"Document text:\n{doc['text'][:cfg.doc_truncate_chars]}"
    )
    out = chat_json(EXTRACTION_SYSTEM, user, model=cfg.model, temperature=0.0)
    phrases = out.get("phrases") or []
    if not isinstance(phrases, list):
        phrases = []
    return {
        "doc_id": doc["doc_id"],
        "doc_summary": str(out.get("doc_summary") or ""),
        "phrases": phrases,
    }


def extract_all(docs: list[dict], profile: dict, cfg: PipelineConfig) -> list[dict]:
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futures = {ex.submit(extract_document, doc, profile, cfg): doc for doc in docs}
        for fut in as_completed(futures):
            doc = futures[fut]
            try:
                rows.append(fut.result())
                print(f"[extract] {doc['doc_id']}")
            except Exception as exc:  # noqa: BLE001
                rows.append({
                    "doc_id": doc["doc_id"],
                    "doc_summary": "",
                    "phrases": [],
                    "error": str(exc),
                })
                print(f"[extract] failed {doc['doc_id']}: {exc}")
    rows.sort(key=lambda r: r["doc_id"])
    return rows


def _kmeans_domains(vectors: np.ndarray, n_domains: int, seed: int) -> list[int]:
    n = len(vectors)
    if n == 0:
        return []
    k = max(1, min(n_domains, n))
    if k == 1:
        return [0] * n
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(
        normalize_rows(vectors.astype(np.float32))
    )
    return [int(x) for x in labels]


def _group_chunk(
    items: list[dict],
    system_prompt: str,
    model: str,
) -> list[list[int]]:
    lines = []
    for i, item in enumerate(items):
        desc = item.get("description") or ""
        title = item.get("title") or item["id"]
        if desc:
            lines.append(f"{i}: {title} -- {desc}")
        else:
            lines.append(f"{i}: {title}")
    out = chat_json(system_prompt, "Items:\n" + "\n".join(lines), model=model, temperature=0.0)
    used: set[int] = set()
    groups: list[list[int]] = []
    for group in out.get("groups") or []:
        members = []
        for member in group.get("members") or []:
            if isinstance(member, int) and 0 <= member < len(items) and member not in used:
                members.append(member)
                used.add(member)
        if len(members) >= 2:
            groups.append(members)
    return groups


def coassociation_groups(
    items: list[dict],
    vectors: np.ndarray,
    *,
    system_prompt: str,
    model: str,
    n_domains: int,
    chunk_size: int,
    rounds: int,
    min_coassoc: float,
    min_meet: int,
    workers: int,
    seed: int,
    fixed_k: int | None = None,
) -> tuple[list[list[str]], np.ndarray]:
    if not items:
        return [], np.zeros((0, 0), dtype=np.float32)
    if len(items) == 1:
        return [[items[0]["id"]]], np.ones((1, 1), dtype=np.float32)

    ids = [item["id"] for item in items]
    id_to_pos = {item_id: i for i, item_id in enumerate(ids)}
    labels = _kmeans_domains(vectors, n_domains, seed)
    by_domain: dict[int, list[int]] = defaultdict(list)
    for pos, label in enumerate(labels):
        by_domain[label].append(pos)

    rng = random.Random(seed)
    chunks: list[list[int]] = []
    for _round in range(rounds):
        for positions in by_domain.values():
            order = positions[:]
            rng.shuffle(order)
            for start in range(0, len(order), chunk_size):
                chunk = order[start:start + chunk_size]
                if len(chunk) >= 2:
                    chunks.append(chunk)

    meet: Counter[tuple[int, int]] = Counter()
    together: Counter[tuple[int, int]] = Counter()
    for chunk in chunks:
        for a, b in combinations(sorted(chunk), 2):
            meet[(a, b)] += 1

    def call(chunk: list[int]) -> list[list[int]]:
        chunk_items = [items[i] for i in chunk]
        local_groups = _group_chunk(chunk_items, system_prompt, model)
        return [[chunk[i] for i in local] for local in local_groups]

    print(f"[group] {len(items)} items, {len(chunks)} LLM chunks")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(call, chunk) for chunk in chunks]
        for fut in as_completed(futures):
            try:
                for group in fut.result():
                    for a, b in combinations(sorted(group), 2):
                        together[(a, b)] += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[group] warning: chunk failed: {exc}")

    n = len(items)
    sim = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(sim, 1.0)
    for (a, b), m in meet.items():
        if m > 0:
            score = together[(a, b)] / m
            sim[a, b] = sim[b, a] = float(score)

    if fixed_k is not None:
        k = max(1, min(fixed_k, n))
        if k == n:
            return [[item_id] for item_id in ids], sim
        from sklearn.cluster import AgglomerativeClustering

        dist = 1.0 - sim
        try:
            clusterer = AgglomerativeClustering(
                n_clusters=k, metric="precomputed", linkage="average"
            )
        except TypeError:
            clusterer = AgglomerativeClustering(
                n_clusters=k, affinity="precomputed", linkage="average"
            )
        cluster_labels = clusterer.fit_predict(dist)
        grouped: dict[int, list[str]] = defaultdict(list)
        for item_id, label in zip(ids, cluster_labels):
            grouped[int(label)].append(item_id)
        groups = list(grouped.values())
        groups.sort(key=lambda g: (-len(g), g[0]))
        return groups, sim

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    for (a, b), m in meet.items():
        if m >= min_meet and sim[a, b] >= min_coassoc:
            union(a, b)

    grouped_pos: dict[int, list[str]] = defaultdict(list)
    for item_id in ids:
        grouped_pos[find(id_to_pos[item_id])].append(item_id)
    groups = list(grouped_pos.values())
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups, sim


def _safe_name(
    system: str,
    user: str,
    model: str,
    fallback: str,
) -> tuple[str, str]:
    try:
        out = chat_json(system, user, model=model, temperature=0.0)
        name = str(out.get("name") or "").strip()
        desc = str(out.get("description") or "").strip()
        return name or fallback, desc
    except Exception as exc:  # noqa: BLE001
        return fallback, f"LLM naming failed: {exc}"


def _support_docs_for_entries(entry_ids: list[str], occurrences: list[dict]) -> list[str]:
    wanted = set(entry_ids)
    return sorted({o["doc_id"] for o in occurrences if o["phrase_entry_id"] in wanted})


def build_topics(
    groups: list[list[str]],
    entries: list[dict],
    occurrences: list[dict],
    phrase_vectors: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[list[dict], np.ndarray, dict[str, str]]:
    entry_by_id = {e["phrase_entry_id"]: e for e in entries}
    rows: list[dict] = []
    entry_to_topic: dict[str, str] = {}

    def make_topic(idx_group: tuple[int, list[str]]) -> dict:
        idx, member_ids = idx_group
        member_entries = [entry_by_id[eid] for eid in member_ids if eid in entry_by_id]
        phrases = [e["text"] for e in member_entries]
        support_docs = _support_docs_for_entries(member_ids, occurrences)
        fallback = phrases[0] if phrases else f"Topic {idx + 1}"
        prompt = (
            "Evidence phrases:\n"
            + "\n".join(f"- {p}" for p in phrases[: cfg.max_name_members])
            + "\n\nSupporting document ids:\n"
            + ", ".join(support_docs[:12])
        )
        name, desc = _safe_name(NAME_TOPIC_SYSTEM, prompt, cfg.model, fallback)
        return {
            "topic_id": f"topic-{idx:04d}",
            "display_title": name,
            "description": desc,
            "member_phrase_entry_ids": member_ids,
            "representative_phrases": phrases[:10],
            "support_doc_ids": support_docs,
            "support_doc_count": len(support_docs),
            "occurrence_count": sum(
                1 for o in occurrences if o["phrase_entry_id"] in set(member_ids)
            ),
        }

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        for topic in ex.map(make_topic, list(enumerate(groups))):
            rows.append(topic)
    rows.sort(key=lambda r: (-r["support_doc_count"], r["topic_id"]))

    vectors: list[np.ndarray] = []
    for topic in rows:
        for entry_id in topic["member_phrase_entry_ids"]:
            entry_to_topic[entry_id] = topic["topic_id"]
        member_rows = [entry_by_id[eid]["embedding_row"] for eid in topic["member_phrase_entry_ids"]]
        if member_rows:
            vectors.append(phrase_vectors[member_rows].mean(axis=0))
        else:
            vectors.append(np.zeros((phrase_vectors.shape[1],), dtype=np.float32))
    return rows, np.asarray(vectors, dtype=np.float32), entry_to_topic


def build_mid_groups(
    groups: list[list[str]],
    topics: list[dict],
    topic_vectors: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[list[dict], np.ndarray]:
    topic_by_id = {t["topic_id"]: t for t in topics}
    row_by_topic = {t["topic_id"]: i for i, t in enumerate(topics)}
    rows: list[dict] = []

    def make_group(idx_group: tuple[int, list[str]]) -> dict:
        idx, topic_ids = idx_group
        members = [topic_by_id[t] for t in topic_ids if t in topic_by_id]
        lines = []
        for topic in members[: cfg.max_name_members]:
            phrases = "; ".join(topic.get("representative_phrases") or [])
            lines.append(f"- {topic['display_title']}: {phrases}")
        fallback = members[0]["display_title"] if members else f"Group {idx + 1}"
        name, desc = _safe_name(NAME_GROUP_SYSTEM, "\n".join(lines), cfg.model, fallback)
        doc_ids = sorted({d for topic in members for d in topic.get("support_doc_ids", [])})
        return {
            "group_id": f"group-{idx:04d}",
            "display_title": name,
            "description": desc,
            "member_topic_ids": topic_ids,
            "support_doc_ids": doc_ids,
            "support_doc_count": len(doc_ids),
        }

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        rows.extend(ex.map(make_group, list(enumerate(groups))))
    rows.sort(key=lambda r: (-r["support_doc_count"], r["group_id"]))

    vectors: list[np.ndarray] = []
    for row in rows:
        member_rows = [row_by_topic[tid] for tid in row["member_topic_ids"] if tid in row_by_topic]
        vectors.append(topic_vectors[member_rows].mean(axis=0))
    return rows, np.asarray(vectors, dtype=np.float32)


def build_aspects(
    groups: list[list[str]],
    mid_groups: list[dict],
    cfg: PipelineConfig,
) -> list[dict]:
    group_by_id = {g["group_id"]: g for g in mid_groups}
    aspects: list[dict] = []

    def make_aspect(idx_group: tuple[int, list[str]]) -> dict:
        idx, group_ids = idx_group
        members = [group_by_id[g] for g in group_ids if g in group_by_id]
        lines = [f"- {g['display_title']}: {g.get('description', '')}" for g in members]
        fallback = members[0]["display_title"] if members else f"Aspect {idx + 1}"
        name, desc = _safe_name(NAME_ASPECT_SYSTEM, "\n".join(lines), cfg.model, fallback)
        topic_ids = [tid for g in members for tid in g.get("member_topic_ids", [])]
        doc_ids = sorted({d for g in members for d in g.get("support_doc_ids", [])})
        return {
            "aspect_id": f"aspect-{idx:02d}",
            "display_title": name,
            "description": desc,
            "member_group_ids": group_ids,
            "member_topic_ids": topic_ids,
            "support_doc_ids": doc_ids,
            "support_doc_count": len(doc_ids),
        }

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        aspects.extend(ex.map(make_aspect, list(enumerate(groups))))
    aspects.sort(key=lambda r: (-r["support_doc_count"], r["aspect_id"]))
    return aspects


def write_dashboard_html(path: Path) -> None:
    html = """<!doctype html>
<meta charset="utf-8">
<title>EviMap POC Artifact</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;line-height:1.45}
.layout{display:grid;grid-template-columns:360px 1fr;gap:24px}
button{display:block;width:100%;text-align:left;margin:4px 0;padding:6px}
mark{background:#ffcf70;padding:1px 3px}
pre{white-space:pre-wrap}
</style>
<h1>EviMap POC Artifact</h1>
<p>This viewer is only for checking the regenerated artifact. The reproducible output is <code>topic_map.json</code>.</p>
<div class="layout"><div id="tree"></div><div id="detail"></div></div>
<script>
async function main(){
  const data = await fetch("topic_map.json").then(r=>r.json());
  const tree = document.getElementById("tree");
  const detail = document.getElementById("detail");
  const topics = new Map(data.topics.map(t=>[t.topic_id,t]));
  const docs = new Map(data.documents.map(d=>[d.doc_id,d]));
  function showTopic(id){
    const t = topics.get(id);
    const occ = data.occurrences.filter(o=>o.topic_id===id).slice(0,20);
    let html = `<h2>${t.display_title}</h2><p>${t.description||""}</p>`;
    html += `<p>${t.support_doc_count} supporting docs</p><h3>Evidence</h3>`;
    for(const o of occ){
      const d = docs.get(o.doc_id);
      const before = d.text.slice(Math.max(0,o.start-90), o.start);
      const hit = d.text.slice(o.start,o.end);
      const after = d.text.slice(o.end, Math.min(d.text.length,o.end+90));
      html += `<p><strong>${o.doc_id}</strong><br>${before}<mark>${hit}</mark>${after}</p>`;
    }
    detail.innerHTML = html;
  }
  for(const a of data.aspects){
    const h = document.createElement("h2");
    h.textContent = a.display_title;
    tree.appendChild(h);
    for(const gid of a.member_group_ids){
      const g = data.mid_groups.find(x=>x.group_id===gid);
      const gh = document.createElement("h3");
      gh.textContent = g.display_title;
      tree.appendChild(gh);
      for(const tid of g.member_topic_ids){
        const t = topics.get(tid);
        const b = document.createElement("button");
        b.textContent = `${t.display_title} (${t.support_doc_count})`;
        b.onclick = ()=>showTopic(tid);
        tree.appendChild(b);
      }
    }
  }
}
main();
</script>
"""
    path.write_text(html, encoding="utf-8")


def write_report(path: Path, cfg: PipelineConfig, counts: dict[str, int]) -> None:
    lines = [
        "# EviMap POC Run Report",
        "",
        f"- evimap_poc_version: `{__version__}`",
        f"- input: `{cfg.input}`",
        f"- llm_model: `{cfg.model}`",
        f"- embedding_backend: `{cfg.embedding_backend}`",
        f"- embedding_model: `{cfg.embedding_model}`",
        f"- documents: **{counts['documents']}**",
        f"- phrase entries: **{counts['phrase_entries']}**",
        f"- matched phrase occurrences: **{counts['occurrences']}**",
        f"- unmatched extracted phrases: **{counts['unmatched']}**",
        f"- leaf topics: **{counts['topics']}**",
        f"- mid groups: **{counts['mid_groups']}**",
        f"- top aspects: **{counts['aspects']}**",
        "",
        "Traceability contract:",
        "",
        "Every occurrence in `03_index/phrase_occurrences.jsonl` and "
        "`04_leaf_topics/topic_occurrences.jsonl` carries `doc_id`, `start`, "
        "`end`, and `matched_text`, so generated labels can be followed back "
        "to exact source spans.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(cfg: PipelineConfig) -> None:
    out = Path(cfg.output)
    ensure_dir(out)
    write_json(out / "config.json", asdict(cfg))

    docs = load_documents(Path(cfg.input), cfg.max_docs)
    write_jsonl(out / "documents.jsonl", docs)
    print(f"[1] loaded {len(docs)} documents")

    print("[2] discovering domain profile with external LLM")
    profile = discover_profile(docs, cfg)
    write_json(out / "01_profile" / "domain_profile.json", profile)

    print("[3] extracting evidence phrases with external LLM")
    extractions = extract_all(docs, profile, cfg)
    write_jsonl(out / "02_extraction" / "extractions.jsonl", extractions)

    print("[4] aligning phrases back to document spans")
    entries, occurrences, unmatched = build_phrase_index(docs, extractions)
    write_jsonl(out / "03_index" / "phrase_entries.jsonl", entries)
    write_jsonl(out / "03_index" / "phrase_occurrences.jsonl", occurrences)
    write_jsonl(out / "03_index" / "unmatched_phrases.jsonl", unmatched)

    print("[5] embedding phrase entries")
    phrase_vectors = embed_texts(
        [e["text"] for e in entries],
        backend=cfg.embedding_backend,
        model_name=cfg.embedding_model,
        base_url=cfg.embedding_base_url,
        batch_size=cfg.embedding_batch_size,
        device=cfg.embedding_device,
    )
    ensure_dir(out / "03_index")
    np.save(out / "03_index" / "phrase_embeddings.npy", phrase_vectors)

    phrase_items = [
        {
            "id": e["phrase_entry_id"],
            "title": e["text"],
            "description": f"docs={e['support_doc_count']}; roles={e['role_hint_dist']}; axes={e['axis_hints_pool']}",
        }
        for e in entries
        if e["occurrence_count"] > 0
    ]
    phrase_rows = [e["embedding_row"] for e in entries if e["occurrence_count"] > 0]
    phrase_vectors_for_grouping = phrase_vectors[phrase_rows]

    print("[6] inducing leaf topics with KMeans-scaffolded LLM co-association")
    ensure_dir(out / "04_leaf_topics")
    leaf_groups, leaf_sim = coassociation_groups(
        phrase_items,
        phrase_vectors_for_grouping,
        system_prompt=FINE_GROUP_SYSTEM,
        model=cfg.model,
        n_domains=cfg.phrase_domains,
        chunk_size=cfg.chunk_size,
        rounds=cfg.rounds,
        min_coassoc=cfg.min_coassoc,
        min_meet=cfg.min_meet,
        workers=cfg.workers,
        seed=cfg.seed,
    )
    np.save(out / "04_leaf_topics" / "phrase_coassociation.npy", leaf_sim)
    topics, topic_vectors, entry_to_topic = build_topics(
        leaf_groups, entries, occurrences, phrase_vectors, cfg
    )
    topic_occurrences = []
    for occ in occurrences:
        topic_id = entry_to_topic.get(occ["phrase_entry_id"])
        if topic_id:
            row = dict(occ)
            row["topic_id"] = topic_id
            topic_occurrences.append(row)
    write_jsonl(out / "04_leaf_topics" / "topics.jsonl", topics)
    write_jsonl(out / "04_leaf_topics" / "topic_occurrences.jsonl", topic_occurrences)

    print("[7] grouping leaf topics into mid-level groups")
    ensure_dir(out / "05_hierarchy")
    topic_items = [
        {
            "id": t["topic_id"],
            "title": t["display_title"],
            "description": "; ".join(t.get("representative_phrases") or []),
        }
        for t in topics
    ]
    mid_id_groups, mid_sim = coassociation_groups(
        topic_items,
        topic_vectors,
        system_prompt=MID_GROUP_SYSTEM,
        model=cfg.model,
        n_domains=cfg.topic_domains,
        chunk_size=cfg.chunk_size,
        rounds=cfg.rounds,
        min_coassoc=cfg.min_coassoc,
        min_meet=cfg.min_meet,
        workers=cfg.workers,
        seed=cfg.seed + 1,
    )
    np.save(out / "05_hierarchy" / "topic_coassociation.npy", mid_sim)
    mid_groups, mid_vectors = build_mid_groups(mid_id_groups, topics, topic_vectors, cfg)
    write_jsonl(out / "05_hierarchy" / "mid_groups.jsonl", mid_groups)

    print("[8] building fixed-k top-level aspects")
    mid_items = [
        {
            "id": g["group_id"],
            "title": g["display_title"],
            "description": g.get("description", ""),
        }
        for g in mid_groups
    ]
    aspect_id_groups, aspect_sim = coassociation_groups(
        mid_items,
        mid_vectors,
        system_prompt=TOP_GROUP_SYSTEM,
        model=cfg.model,
        n_domains=max(1, min(cfg.topic_domains, len(mid_items))),
        chunk_size=cfg.chunk_size,
        rounds=cfg.rounds,
        min_coassoc=cfg.min_coassoc,
        min_meet=cfg.min_meet,
        workers=cfg.workers,
        seed=cfg.seed + 2,
        fixed_k=cfg.top_k,
    )
    np.save(out / "05_hierarchy" / "mid_group_coassociation.npy", aspect_sim)
    aspects = build_aspects(aspect_id_groups, mid_groups, cfg)
    write_jsonl(out / "05_hierarchy" / "aspects.jsonl", aspects)

    print("[9] writing compact artifact")
    artifact = {
        "config": asdict(cfg),
        "profile": profile,
        "documents": docs,
        "phrase_entries": entries,
        "topics": topics,
        "mid_groups": mid_groups,
        "aspects": aspects,
        "occurrences": topic_occurrences,
    }
    write_json(out / "06_artifact" / "topic_map.json", artifact)
    write_dashboard_html(out / "06_artifact" / "dashboard.html")
    write_report(
        out / "06_artifact" / "run_report.md",
        cfg,
        {
            "documents": len(docs),
            "phrase_entries": len(entries),
            "occurrences": len(occurrences),
            "unmatched": len(unmatched),
            "topics": len(topics),
            "mid_groups": len(mid_groups),
            "aspects": len(aspects),
        },
    )
    print(f"[done] wrote {out}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the full EviMap POC pipeline.")
    p.add_argument("--input", required=True, help="JSONL documents: doc_id, text, metadata")
    p.add_argument("--output", required=True, help="run output directory")
    p.add_argument("--model", default=DEFAULT_LLM_MODEL)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--profile-sample-size", type=int, default=8)
    p.add_argument("--doc-truncate-chars", type=int, default=12000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=20)
    p.add_argument("--min-coassoc", type=float, default=0.67)
    p.add_argument("--min-meet", type=int, default=1)
    p.add_argument("--phrase-domains", type=int, default=8)
    p.add_argument("--topic-domains", type=int, default=4)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--embedding-backend", choices=["local", "openai", "remote"], default=DEFAULT_BACKEND)
    p.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--embedding-base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--embedding-batch-size", type=int, default=128)
    p.add_argument("--embedding-device", default=DEFAULT_DEVICE)
    p.add_argument("--max-name-members", type=int, default=18)
    return p


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(**vars(args))
