#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
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


def load_stats(artifact_dir: Path) -> dict:
    topic_map = json.loads((artifact_dir / "topic_map.json").read_text(encoding="utf-8"))
    config = topic_map.get("config") or {}
    return {
        "title": config.get("output", "Generated Dashboard"),
        "n_docs": len(topic_map.get("documents") or []),
        "n_topics": len(topic_map.get("topics") or []),
        "n_aspects": len(topic_map.get("aspects") or []),
        "n_spans": len(topic_map.get("occurrences") or []),
    }


def landing_page(
    title: str,
    stats: dict,
    *,
    contact_email: str,
    github_url: str,
) -> str:
    safe_title = html.escape(title)
    docs = int(stats.get("n_docs") or 0)
    topics = int(stats.get("n_topics") or 0)
    aspects = int(stats.get("n_aspects") or 0)
    spans = int(stats.get("n_spans") or 0)
    contact_href = (
        f"mailto:{html.escape(contact_email)}"
        "?subject=EviMap%3A%20add%20a%20new%20corpus"
    )
    safe_github_url = html.escape(github_url, quote=True)

    cards = [
        (
            "dashboard.html",
            "Open generated dashboard",
            f"{docs:,} documents, {topics:,} topics, {spans:,} evidence spans",
            "",
            ' id="card-dashboard" data-domain-card="true"',
        ),
        (
            "topic_map.json",
            "Inspect topic map JSON",
            f"{aspects:,} top-level aspects with span-level provenance",
            "",
            ' data-domain-card="true"',
        ),
        (
            "run_report.md",
            "Read run report",
            "Pipeline settings, output counts, and traceability contract",
            "",
            ' data-domain-card="true"',
        ),
        (
            contact_href,
            "+ Add a new corpus",
            "Don't see your domain? Contact the author to map a new corpus.",
            " card-action",
            ' data-domain-card="true"',
        ),
        (
            safe_github_url,
            "Run it yourself",
            "Get the open-source code on GitHub and map your own corpus.",
            " card-action",
            ' data-domain-card="true"',
        ),
    ]
    card_html = []
    for href, heading, body, cls, extra_attrs in cards:
        attrs = ' target="_blank" rel="noopener"' if href.startswith("http") else ""
        card_html.append(
            f'<a class="card{cls}" href="{href}"{attrs}{extra_attrs}>'
            f"<h2>{html.escape(heading)}</h2>"
            f"<p>{html.escape(body)}</p></a>"
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}: Evidence-Grounded Topic Maps</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; color: #1c1f24; background: #f7f7f9; line-height: 1.5;
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                      Helvetica, Arial, sans-serif; }}
  .wrap {{ max-width: 900px; margin: 0 auto; padding: 0 24px; }}
  .topbar {{ background: #1c2733; color: #f7f7f9; padding: 30px 0 28px; }}
  .topbar h1 {{ margin: 0; font-size: 27px; letter-spacing: -0.02em; font-weight: 750;
               background: linear-gradient(120deg, #22d3ee 0%, #6366f1 48%, #ec4899 100%);
               -webkit-background-clip: text; background-clip: text;
               -webkit-text-fill-color: transparent; color: transparent; }}
  .topbar p {{ margin: 9px 0 0; font-size: 14px; color: #aeb6bf; max-width: 660px; }}
  .hero-actions {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }}
  .hero-action {{ display: inline-flex; align-items: center; min-height: 34px;
                 border-radius: 6px; padding: 7px 12px; text-decoration: none;
                 font-size: 13px; font-weight: 600; border: 0; cursor: pointer;
                 font-family: inherit; }}
  .hero-action-primary {{ background: #2dd4bf; color: #063b36; }}
  .hero-action:hover {{ filter: brightness(1.04); }}
  main {{ padding: 34px 0 64px; }}
  .section-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: .07em;
                   color: #6b7380; margin: 0 0 15px; font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px,1fr));
          gap: 14px; grid-auto-rows: 1fr; }}
  .card {{ display: flex; flex-direction: column; justify-content: center;
          min-height: 92px; background: #fff; border: 1px solid #e1e3e7;
          border-radius: 10px; padding: 16px 18px; text-decoration: none; color: inherit;
          transition: border-color .15s, box-shadow .15s, transform .15s; }}
  .card:hover {{ border-color: #f7a23b; box-shadow: 0 3px 14px rgba(28,39,51,.08);
                transform: translateY(-1px); }}
  .card h2 {{ font-size: 17px; margin: 0 0 6px; font-weight: 600; }}
  .card p {{ margin: 0; color: #6b7380; font-size: 13px; }}
  .card-action {{ border-style: dashed; background: #fcfcfd; }}
  .card-action h2 {{ color: #4d8db4; }}
  .card-action:hover {{ border-color: #4d8db4; }}
  .home-tour-note {{ display: none; position: fixed; z-index: 3; width: min(340px, calc(100vw - 32px));
                    margin: 0; padding: 10px 12px;
                    border: 1px solid #99f6e4; border-left: 4px solid #14b8a6;
                    border-radius: 8px; background: #ecfeff; color: #134e4a;
                    box-shadow: 0 14px 32px rgba(17,24,39,.18);
                    font-size: 13px; line-height: 1.45; }}
  .home-tour-note::before {{ content: ""; position: absolute; left: 50%; top: auto; bottom: -7px;
                            width: 12px; height: 12px; background: #ecfeff;
                            border-left: 1px solid #99f6e4; border-bottom: 1px solid #99f6e4;
                            transform: translateX(-50%) rotate(-45deg); }}
  .home-tour-note.note-below::before {{ left: 28px; top: -7px; bottom: auto; transform: rotate(135deg); }}
  body.home-tour-active.home-tour-note-ready .home-tour-note,
  body.home-explore-active.home-tour-note-ready .home-tour-note {{ display: block; }}
  body.home-tour-active::before,
  body.home-explore-active::before {{
    content: ""; position: fixed; inset: 0; z-index: 1;
    background: rgba(12,18,28,.18); pointer-events: none;
  }}
  body.home-tour-active .card[data-domain-card="true"],
  body.home-explore-active .card[data-domain-card="true"] {{
    position: relative; z-index: 2; border-color: #14b8a6; background: #ecfeff;
    outline: 3px solid rgba(45,212,191,.30); outline-offset: 3px;
    box-shadow: 0 14px 34px rgba(17,24,39,.18);
  }}
  body.home-tour-active #card-dashboard {{
    outline: 4px solid rgba(45,212,191,.34); outline-offset: 4px;
    box-shadow: 0 18px 42px rgba(17,24,39,.22);
    animation: home-tour-pulse 1.6s ease-in-out infinite;
  }}
  @keyframes home-tour-pulse {{
    0%, 100% {{ transform: translateY(-1px); }}
    50% {{ transform: translateY(-4px); }}
  }}
</style></head>
<body>
  <header class="topbar">
    <div class="wrap">
      <h1>{safe_title}</h1>
      <p>Evidence-grounded topic maps. Browse an unfamiliar corpus as a topic
      map. Every topic traces back to evidence phrases highlighted in the
      original documents.</p>
      <div class="hero-actions">
        <button class="hero-action hero-action-primary" id="start-home-tour" type="button">Start guided tour</button>
      </div>
    </div>
  </header>
  <main>
    <div class="wrap">
      <p class="home-tour-note" id="home-tour-note">Start with the generated dashboard to inspect the evidence-grounded topic map. After that, use the other cards to inspect artifacts, contact the authors, or run EviMap yourself.</p>
      <p class="section-label">Open this run</p>
      <div class="grid">
        {"".join(card_html)}
      </div>
    </div>
  </main>
  <script>
    const startTour = document.getElementById("start-home-tour");
    const primaryCard = document.getElementById("card-dashboard");
    const note = document.getElementById("home-tour-note");
    const cards = Array.from(document.querySelectorAll('.card[data-domain-card="true"]'));
    let homeGuideMode = "tour";
    if (startTour && primaryCard && note) {{
      const activeCard = () => homeGuideMode === "explore" ? cards[0] : primaryCard;
      const placeNote = () => {{
        const active = document.body.classList.contains("home-tour-active") ||
                       document.body.classList.contains("home-explore-active");
        if (!active) return;
        document.body.classList.remove("home-tour-note-ready");
        const targetCard = activeCard();
        if (!targetCard) return;
        const rect = targetCard.getBoundingClientRect();
        const gap = 18;
        note.style.visibility = "hidden";
        note.style.display = "block";
        const noteW = Math.min(340, window.innerWidth - 32);
        const noteH = note.offsetHeight || 72;
        note.classList.remove("note-below");
        let left = rect.left + (rect.width - noteW) / 2;
        let top = rect.top - noteH - gap;
        if (top < 16) {{
          top = rect.bottom + gap;
          note.classList.add("note-below");
        }}
        left = Math.max(16, Math.min(window.innerWidth - noteW - 16, left));
        note.style.left = `${{left}}px`;
        note.style.top = `${{Math.max(16, Math.min(window.innerHeight - noteH - 16, top))}}px`;
        note.style.display = "";
        note.style.visibility = "";
        document.body.classList.add("home-tour-note-ready");
      }};
      const startHomeGuide = (mode) => {{
        homeGuideMode = mode;
        document.body.classList.toggle("home-tour-active", mode === "tour");
        document.body.classList.toggle("home-explore-active", mode === "explore");
        document.body.classList.remove("home-tour-note-ready");
        note.textContent = mode === "tour"
          ? "Start with the generated dashboard to inspect the evidence-grounded topic map."
          : "Pick any card: revisit the dashboard, inspect the JSON/report, contact the authors, or run EviMap yourself.";
        const targetCard = activeCard();
        if (targetCard) targetCard.scrollIntoView({{ block: "center", behavior: "smooth" }});
        window.setTimeout(() => {{
          if (targetCard) targetCard.focus({{ preventScroll: true }});
          placeNote();
        }}, 260);
      }};
      startTour.addEventListener("click", () => startHomeGuide("tour"));
      window.addEventListener("resize", placeNote);
      window.addEventListener("scroll", placeNote, {{ passive: true }});
      document.addEventListener("keydown", ev => {{
        if (ev.key === "Escape") document.body.classList.remove("home-tour-active", "home-explore-active", "home-tour-note-ready");
      }});
      const params = new URLSearchParams(window.location.search);
      if (params.get("explore") === "1") {{
        window.setTimeout(() => startHomeGuide("explore"), 160);
      }}
    }}
  </script>
</body></html>
"""


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
            "home": "/index.html",
            "viewer": "/dashboard.html",
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
    parser.add_argument("--contact-email", default="zhiyin.tan@l3s.de")
    parser.add_argument("--github-url", default="https://github.com/Fireblossom/EviMap")
    args = parser.parse_args()

    root = Path.cwd()
    run_dir = (root / args.run).resolve() if not Path(args.run).is_absolute() else Path(args.run)
    out_dir = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    artifact_dir = run_dir / "06_artifact"

    copy_required(artifact_dir, out_dir)
    stats = load_stats(artifact_dir)
    (out_dir / "index.html").write_text(
        landing_page(
            args.title,
            stats,
            contact_email=args.contact_email,
            github_url=args.github_url,
        ),
        encoding="utf-8",
    )
    write_pages_files(out_dir, run_dir, args.title)
    print(f"Packaged frontend: {artifact_dir} -> {out_dir}")
    print(f"Open locally: python scripts/serve_debug_site.py --dir {out_dir} --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
