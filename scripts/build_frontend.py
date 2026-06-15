#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REQUIRED = [
    "topic_map.json",
    "dashboard.html",
    "run_report.md",
]


def copy_required(artifact_dir: Path, out_dir: Path) -> None:
    missing = [name for name in REQUIRED if not (artifact_dir / name).exists()]
    if missing:
        names = ", ".join(missing)
        raise SystemExit(
            f"missing frontend artifact(s): {names}\n"
            f"Expected a completed pipeline run with {artifact_dir}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for item in out_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    shutil.copy2(artifact_dir / "topic_map.json", out_dir / "topic_map.json")
    shutil.copy2(artifact_dir / "run_report.md", out_dir / "run_report.md")
    shutil.copy2(artifact_dir / "dashboard.html", out_dir / "dashboard.html")
    shutil.copy2(artifact_dir / "dashboard.html", out_dir / "index.html")


def write_pages_files(out_dir: Path, run_dir: Path, title: str) -> None:
    headers = """/*
  X-Content-Type-Options: nosniff

/index.html
  Cache-Control: no-cache

/dashboard.html
  Cache-Control: no-cache

/topic_map.json
  Cache-Control: public, max-age=300

/run_report.md
  Cache-Control: public, max-age=300
"""
    redirects = """/dashboard /dashboard.html 200
/report /run_report.md 200
"""
    manifest = {
        "title": title,
        "source_run": str(run_dir),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "entrypoints": {
            "viewer": "/index.html",
            "topic_map": "/topic_map.json",
            "run_report": "/run_report.md",
        },
    }
    (out_dir / "_headers").write_text(headers, encoding="utf-8")
    (out_dir / "_redirects").write_text(redirects, encoding="utf-8")
    (out_dir / "deploy_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package an EviMap pipeline run for static frontend deployment."
    )
    parser.add_argument(
        "--run",
        default="runs/sample_job_posts",
        help="Pipeline run directory containing 06_artifact/",
    )
    parser.add_argument(
        "--out",
        default="dist",
        help="Static output directory for Cloudflare Pages or any static host.",
    )
    parser.add_argument("--title", default="EviMap")
    args = parser.parse_args()

    root = Path.cwd()
    run_dir = (root / args.run).resolve() if not Path(args.run).is_absolute() else Path(args.run)
    out_dir = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    artifact_dir = run_dir / "06_artifact"

    copy_required(artifact_dir, out_dir)
    write_pages_files(out_dir, run_dir, args.title)
    print(f"Packaged frontend: {artifact_dir} -> {out_dir}")
    print(f"Open locally: python -m http.server 8000 --directory {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
