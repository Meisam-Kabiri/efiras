"""
amendment_checker.py

AmendmentChecker: re-fetches each Bucket A document's live EUR-Lex URL and
compares it against the cached html_cache/*.html used at chunk time, to flag
documents whose source has changed since they were last chunked.

Scope: HTML-sourced documents only (BUCKET_A entries with format == "html").
PDF-sourced documents -- SOLVENCY2_L2 (nominally Bucket A but served as PDF)
and all of BUCKET_B -- are skipped on purpose. Those don't have a cheap,
stable "live page" to diff against the same way: re-checking them means
re-running the paid LLM extraction pipeline (pdf_to_html.py), not an HTTP
fetch + text compare. This class only tells you "the HTML changed, go
re-chunk it" -- it does not re-chunk or re-embed anything itself.

Needs:  pip install requests beautifulsoup4
"""

import hashlib
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from buckets_config import BUCKET_A

HTML_CACHE_DIR = Path("html_cache")


class AmendmentChecker:
    """
    Checks a set of (doc_id, name, url, format) entries for source changes:
    fetches each url fresh, cleans it the same way on both sides, and hashes
    the result against the cached html_cache/<doc_id>.html used at chunk
    time. A hash mismatch means the live document text differs from what was
    chunked -- i.e. an amendment (or any other edit) landed since last fetch.
    """

    def __init__(self, docs=None, cache_dir=HTML_CACHE_DIR, request_timeout=120):
        self.docs = docs if docs is not None else [d for d in BUCKET_A if d[3] == "html"]
        self.cache_dir = Path(cache_dir)
        self.request_timeout = request_timeout

    @staticmethod
    def _clean_text(html_text):
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _hash(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _fetch_live_html(self, url):
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=self.request_timeout)
        r.raise_for_status()
        if len(r.text) < 10_000:
            raise RuntimeError(
                f"suspiciously short response ({len(r.text)} chars, status {r.status_code}) "
                "-- likely a bot-block / JS-only shell, not a real empty document"
            )
        return r.text

    def check_one(self, doc_id, name, url, fmt):
        cache_file = self.cache_dir / f"{doc_id.lower()}.html"
        if not cache_file.exists():
            return {"doc_id": doc_id, "status": "no_cache",
                     "detail": f"{cache_file} not found -- never chunked"}

        cached_hash = self._hash(self._clean_text(cache_file.read_text(encoding="utf-8")))

        try:
            live_html = self._fetch_live_html(url)
        except (requests.RequestException, RuntimeError) as e:
            return {"doc_id": doc_id, "status": "fetch_error", "detail": str(e)}

        live_hash = self._hash(self._clean_text(live_html))

        if live_hash == cached_hash:
            return {"doc_id": doc_id, "status": "unchanged"}

        return {"doc_id": doc_id, "status": "amended",
                 "detail": f"live text no longer matches {cache_file}",
                 "live_html": live_html}

    def check_all(self, verbose=True):
        results = []
        for doc_id, name, url, fmt in self.docs:
            if verbose:
                print(f"Checking {doc_id}...")
            result = self.check_one(doc_id, name, url, fmt)
            results.append(result)
            if verbose:
                print(f"  -> {result['status']}")
        return results


if __name__ == "__main__":
    checker = AmendmentChecker()
    results = checker.check_all()

    amended = [r for r in results if r["status"] == "amended"]
    errors = [r for r in results if r["status"] == "fetch_error"]
    no_cache = [r for r in results if r["status"] == "no_cache"]
    unchanged = [r for r in results if r["status"] == "unchanged"]

    print(f"\n{len(unchanged)} unchanged, {len(amended)} amended, "
          f"{len(errors)} fetch errors, {len(no_cache)} never chunked "
          f"-- {len(results)} checked.")
    for r in amended:
        print(f"  AMENDED: {r['doc_id']} -- {r['detail']}")
    for r in errors:
        print(f"  FETCH ERROR: {r['doc_id']} -- {r['detail']}")
    for r in no_cache:
        print(f"  NO CACHE: {r['doc_id']} -- {r['detail']}")
