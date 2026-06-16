#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from build_frontend import landing_page, load_stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a one-port local debug site for an EviMap run."
    )
    parser.add_argument(
        "--run",
        default="runs/sample_job_posts",
        help="Pipeline run directory containing 06_artifact/",
    )
    parser.add_argument(
        "--out",
        default="site/debug",
        help="Debug site directory with a homepage and symlinked artifact.",
    )
    parser.add_argument("--title", default="EviMap")
    parser.add_argument("--contact-email", default="zhiyin.tan@l3s.de")
    parser.add_argument("--github-url", default="https://github.com/Fireblossom/EviMap")
    args = parser.parse_args()

    root = Path.cwd()
    run_dir = (root / args.run).resolve() if not Path(args.run).is_absolute() else Path(args.run)
    artifact_dir = run_dir / "06_artifact"
    out_dir = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    if not (artifact_dir / "dashboard.html").exists():
        raise SystemExit(f"missing completed artifact directory: {artifact_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    run_link = out_dir / "run"
    if run_link.is_symlink() or run_link.exists():
        if run_link.is_dir() and not run_link.is_symlink():
            shutil.rmtree(run_link)
        else:
            run_link.unlink()
    run_link.symlink_to(artifact_dir)

    stats = load_stats(artifact_dir)
    html = landing_page(
        args.title,
        stats,
        contact_email=args.contact_email,
        github_url=args.github_url,
    ).replace('href="dashboard.html"', 'href="run/dashboard.html"')
    html = html.replace('href="topic_map.json"', 'href="run/topic_map.json"')
    html = html.replace('href="run_report.md"', 'href="run/run_report.md"')
    out_dir.joinpath("index.html").write_text(html, encoding="utf-8")

    print(f"site/debug ready: {out_dir}")
    print(f"  cd {out_dir} && python -m http.server 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
