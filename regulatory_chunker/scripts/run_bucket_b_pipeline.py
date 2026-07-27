"""
run_bucket_b_pipeline.py

Runs the Bucket B (PDF) pipeline for every configured document: extracts
structural events via the LLM (pdf_to_html.run -- the only stage that costs
money) and translates them into TreeChunker-consumable HTML
(pdf_html_builder.events_to_html). Any doc whose html_cache/ output already
exists is skipped entirely -- no LLM call, no cost -- so this is safe to
rerun any time new documents get added to DOCS below without ever
re-processing ones already done. Each doc's raw events are also saved to
events_cache/, so scripts/build_html_from_events.py can rebuild/tweak the
HTML afterwards for free, without ever re-extracting.

Usage:  python scripts/run_bucket_b_pipeline.py   (run from the repo root)
Needs:  ANTHROPIC_API_KEY set in .env -- this makes real, billed LLM calls
        for any document not already in html_cache/.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdf_to_html import run
from pdf_html_builder import events_to_html

MODEL = "claude-sonnet-5"
MAX_BATCHES = None  # None = whole document; keep this small while testing

DOCS = [
    # (pdf_path, doc_id, html_out_file)
    ("flat_cache/cssf_18_698.pdf", "CSSF_18_698", "html_cache/cssf_18_698.html"),
    ("flat_cache/basel_iii.pdf", "BASEL3", "html_cache/basel_iii.html"),
    ("flat_cache/fatf_2012.pdf", "FATF2012", "html_cache/fatf_2012.html"),
]

if __name__ == "__main__":
    for pdf_path, doc_id, html_out in DOCS:
        html_out_path = Path(html_out)
        if html_out_path.exists():
            print(f"skip {doc_id}: {html_out_path} already exists")
            continue

        scope = f"first {MAX_BATCHES} batches" if MAX_BATCHES else "full document"
        print(f"\nextracting structure from {pdf_path} via {MODEL} ({scope})")

        events, final_stack, known_types = run(
            pdf_path=pdf_path,
            doc_id=doc_id,
            model=MODEL,
            max_batches=MAX_BATCHES,
        )

        if final_stack:
            still_open = ", ".join(f"{s['type']}:{s['label']}" for s in final_stack)
            print(
                f"NOTE: {len(final_stack)} structural level(s) still open at the end "
                f"of this run (expected when MAX_BATCHES cuts the document short): {still_open}"
            )

        events_out_path = Path(f"events_cache/{doc_id.lower()}_events.json")
        events_out_path.parent.mkdir(parents=True, exist_ok=True)
        events_out_path.write_text(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "model": MODEL,
                    "events": events,
                    "final_stack": final_stack,
                    "known_types": known_types,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"{len(events)} total events, {len(known_types)} known types -> {events_out_path}")

        html = events_to_html(events, doc_id=doc_id)
        html_out_path.parent.mkdir(parents=True, exist_ok=True)
        html_out_path.write_text(html, encoding="utf-8")
        print(f"wrote {len(html):,} chars of HTML -> {html_out_path}")
