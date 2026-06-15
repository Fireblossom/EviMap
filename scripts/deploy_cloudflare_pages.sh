#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROJECT="${1:-${CF_PAGES_PROJECT:-evimap-poc}}"
RUN_DIR="${RUN_DIR:-runs/sample_job_posts}"
DIST_DIR="${DIST_DIR:-dist}"
BRANCH="${CF_PAGES_BRANCH:-main}"
TITLE="${EVIMAP_SITE_TITLE:-EviMap POC}"

if ! command -v npx >/dev/null 2>&1; then
  echo "ERROR: npx is required for Wrangler. Install Node.js first." >&2
  exit 1
fi

if [ -z "${CLOUDFLARE_API_TOKEN:-}" ]; then
  echo "ERROR: set CLOUDFLARE_API_TOKEN before deploying." >&2
  echo "Token permission should allow Cloudflare Pages read/write for the target account." >&2
  exit 1
fi

python scripts/build_frontend.py \
  --run "$RUN_DIR" \
  --out "$DIST_DIR" \
  --title "$TITLE"

echo "Deploying $DIST_DIR to Cloudflare Pages project '$PROJECT' on branch '$BRANCH' ..."
npx wrangler pages deploy "$DIST_DIR" \
  --project-name="$PROJECT" \
  --branch="$BRANCH"

