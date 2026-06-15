#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

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
