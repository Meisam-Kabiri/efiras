"""
build_html_from_events.py

Stage 2 of the Bucket B pipeline: reads a saved event list (from
scripts/run_bucket_b_pipeline.py, cached in events_cache/) and builds HTML
from it via pdf_html_builder.events_to_html. Free, local, no LLM calls --
safe to rerun as many times as needed while iterating on the translator
itself, without ever re-extracting (that's the separate, expensive stage 1
step).

Optional -- run_bucket_b_pipeline.py already does this as part of its own
extract+save+build sequence for any document it processes. This script is
only useful on its own if you've changed pdf_html_builder.py's translation
logic and want to rebuild a document's HTML from events already saved to
events_cache/, without re-paying for LLM extraction.

Usage:  python scripts/build_html_from_events.py   (run from the repo root)
Needs:  events_cache/<doc>_events.json to already exist.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdf_html_builder import events_to_html

EVENTS_FILE = "events_cache/cssf_18_698_events.json"
OUT_FILE = "html_cache/cssf_18_698.html"

if __name__ == "__main__":
    data = json.loads(Path(EVENTS_FILE).read_text(encoding="utf-8"))
    html = events_to_html(data["events"], doc_id=data["doc_id"])

    out_path = Path(OUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    print(f"{len(data['events'])} events -> {len(html):,} chars of HTML -> {out_path}")
