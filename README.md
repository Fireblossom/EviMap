# EviMap POC

## Live Demo

https://main.evimap-demo.pages.dev/

This folder is a GitHub-ready proof of concept for the full EviMap pipeline
described in the paper:

```text
documents
  -> LLM evidence phrase extraction
  -> span alignment and phrase index
  -> local sentence-transformer embeddings
  -> KMeans-scaffolded, LLM co-association grouping
  -> leaf topics, mid-level groups, top-level aspects
  -> auditable topic-map artifacts
```

The static dashboard is optional. Reproduction is centered on regenerating the
pipeline artifacts from input documents, not on downloading a prebuilt web page.

## 1. Install

Use Python 3.9+.

```bash
cd evimap_poc
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure the external LLM

EviMap uses an OpenAI-compatible chat-completions API for evidence extraction,
semantic grouping, and naming. The defaults match the paper setup: a local
OpenAI-compatible DeepSeek endpoint.

```bash
cp .env.example .env
export EVIMAP_LLM_BASE_URL=http://127.0.0.1:18021/v1
export EVIMAP_LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash
export EVIMAP_LLM_API_KEY=local
```

For OpenAI-hosted models, leave `EVIMAP_LLM_BASE_URL` empty or unset and set
`OPENAI_API_KEY`. Do not set `EVIMAP_LLM_ENABLE_THINKING` unless your
OpenAI-compatible backend accepts that extension field.

## 3. Run the full pipeline

```bash
python -m evimap.run_pipeline \
  --input data/sample_job_posts.jsonl \
  --output runs/sample_job_posts \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --embedding-backend local \
  --embedding-model paraphrase-multilingual-MiniLM-L12-v2 \
  --top-k 14 \
  --phrase-domains 12 \
  --topic-domains 4 \
  --rounds 5 \
  --chunk-size 60 \
  --min-coassoc 0.7 \
  --workers 64
```

Equivalent convenience command:

```bash
bash scripts/run_sample.sh
```

Both commands call the configured external LLM. The POC does not replace the
paper pipeline with a mock model.

The parameters above mirror the paper-style configuration: DeepSeek-V4-Flash
for LLM calls, `paraphrase-multilingual-MiniLM-L12-v2` for phrase embeddings,
five voting rounds, a 0.7 co-association threshold, and a fixed 14-aspect top
layer. The sample corpus is intentionally small so that the full external-LLM
pipeline can be tested quickly. For a real corpus, provide JSONL records with
this shape:

```json
{"doc_id": "doc-001", "text": "full document text", "metadata": {"source": "optional"}}
```

## 4. Outputs

The run directory is designed for audit and debugging:

```text
runs/sample_job_posts/
  config.json
  documents.jsonl
  01_profile/domain_profile.json
  02_extraction/extractions.jsonl
  03_index/phrase_entries.jsonl
  03_index/phrase_occurrences.jsonl
  03_index/unmatched_phrases.jsonl
  03_index/phrase_embeddings.npy
  04_leaf_topics/topics.jsonl
  04_leaf_topics/topic_occurrences.jsonl
  05_hierarchy/mid_groups.jsonl
  05_hierarchy/aspects.jsonl
  06_artifact/topic_map.json
  06_artifact/run_report.md
  06_artifact/dashboard.html
```

Important files:

- `phrase_occurrences.jsonl`: every matched evidence phrase with `doc_id`,
  `start`, and `end` character offsets.
- `topics.jsonl`: leaf topics induced from evidence phrases. Each topic keeps
  member phrase ids and supporting document ids.
- `aspects.jsonl`: top-level aspects with nested mid groups and leaf topic ids.
- `topic_map.json`: compact artifact for downstream inspection or a dashboard.

## 5. Optional dashboard check

The generated dashboard is only a lightweight artifact viewer.

```bash
python -m http.server 8000 --directory runs/sample_job_posts/06_artifact
```

Open `http://localhost:8000/dashboard.html`.

## 6. Package and deploy the frontend

After a pipeline run finishes, package its generated artifact for a static host:

```bash
python scripts/build_frontend.py \
  --run runs/sample_job_posts \
  --out dist
```

This writes:

```text
dist/
  index.html
  dashboard.html
  topic_map.json
  run_report.md
  deploy_manifest.json
  _headers
  _redirects
```

Preview the packaged frontend:

```bash
python -m http.server 8000 --directory dist
```

Deploy the packaged frontend to Cloudflare Pages with Wrangler direct upload:

```bash
export CLOUDFLARE_API_TOKEN=...
bash scripts/deploy_cloudflare_pages.sh evimap-poc
```

Useful environment variables:

```bash
export RUN_DIR=runs/sample_job_posts
export DIST_DIR=dist
export CF_PAGES_BRANCH=main
export EVIMAP_SITE_TITLE="EviMap POC"
```

Equivalent npm scripts are included for convenience:

```bash
npm run build:frontend
npm run deploy:pages -- evimap-poc
```

## 7. Tests

The included tests do not call the LLM.

```bash
PYTHONPATH=. python -m unittest discover -s tests
```

## 8. Notes for larger corpora

- Increase `--workers` only as far as your LLM endpoint can handle.
- Increase `--phrase-domains` and keep `--chunk-size` modest so each LLM
  grouping call sees a small, related context.
- Use `--max-docs` for a paper/demo sample before running a full corpus.
- The embedding default is `paraphrase-multilingual-MiniLM-L12-v2`, matching
  the paper. You can switch to an OpenAI-compatible embedding endpoint with
  `--embedding-backend openai`.
