#!/usr/bin/env python3
"""
Estimate the typical (body-paragraph) number of characters per line in a PDF.

Heuristics:
- Determine body text by finding the modal font size across all spans (weighted by char count).
- Keep lines whose weighted-average span size is close to that body size.
- Remove invisible/control characters (zero-width, BOM, etc.) and normalize whitespace.
- Skip very short lines and obvious bullets.

Note:
- This won’t work on scanned PDFs without OCR.
- Tweak thresholds (MIN_LINE_LEN, SIZE_TOLERANCE) if needed.
"""

import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import fitz  # PyMuPDF

# --- Tunables ---
MIN_LINE_LEN = 25  # ignore short/bullet-ish lines
SIZE_TOLERANCE = 0.10  # ±10% around body font size
ROUND_SIZE = 1  # round font sizes to 0.1pt (1 = 0.1, 0 = 1.0, etc.)

# --- New tunables for dynamic thresholding ---
MIN_LEN_FLOOR = 20  # never go below this
MIN_OCCUPANCY_RATIO = 0.7  # e.g., keep lines with >= 50% of expected full-line chars
WIDTH_QUANTILE = (
    0.75  # use the 75th percentile of line widths as "typical column width"
)

# Common bullet-like starters to strip for length decisions (kept if inside text)
BULLET_PREFIX_RE = re.compile(
    r"^\s*(?:[\-\u2022\u2023\u25E6\u2043\u2219\u00B7•◦▪‣]+|\(?\d+\)?\.?|[A-Za-z]\)|\d{1,2}\.)\s+"
)

# Invisible / zero-width chars to drop outright
INVISIBLE_CODEPOINTS = {
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\ufeff",  # zero width no-break space (BOM)
}


def clean_text(s: str) -> str:
    """Remove invisible/control chars, normalize whitespace; keep normal spaces."""
    # drop specific invisible codepoints
    s = "".join(ch for ch in s if ch not in INVISIBLE_CODEPOINTS)
    # drop all Unicode control/format chars (Cc, Cf)
    s = "".join(ch for ch in s if unicodedata.category(ch) not in ("Cc", "Cf"))
    # normalize whitespace to single spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def weighted_mode_font_size(doc) -> float:
    """Find modal font size across spans, weighted by cleaned char count."""
    size_weights = defaultdict(int)
    for page in doc:
        pd = page.get_text("dict")
        for b in pd.get("blocks", []):
            if b.get("type", 0) != 0:
                continue  # non-text
            for ln in b.get("lines", []):
                for sp in ln.get("spans", []):
                    txt = clean_text(sp.get("text", ""))
                    if not txt:
                        continue
                    size = round(sp.get("size", 0.0), ROUND_SIZE)
                    size_weights[size] += len(txt)
    if not size_weights:
        return 0.0
    # modal by weight
    return max(size_weights.items(), key=lambda kv: kv[1])[0]


def line_weighted_size(line) -> float:
    """Weighted average font size for a PyMuPDF line (dict)."""
    total_chars = 0
    total = 0.0
    for sp in line.get("spans", []):
        txt = clean_text(sp.get("text", ""))
        n = len(txt)
        if n == 0:
            continue
        total_chars += n
        total += n * sp.get("size", 0.0)
    if total_chars == 0:
        return 0.0
    return total / total_chars


def _quantile(values, q):
    """Simple quantile without numpy (q in [0,1])."""
    if not values:
        return 0.0
    vals = sorted(values)
    if q <= 0:
        return vals[0]
    if q >= 1:
        return vals[-1]
    import math

    idx = max(0, min(len(vals) - 1, math.ceil(q * len(vals)) - 1))
    return vals[idx]


def estimate_dynamic_min_len(doc, body_size, size_tol=0.10):
    """Return (dynamic_min_len, expected_capacity, avg_char_w, typical_col_w)."""
    low, high = body_size * (1 - size_tol), body_size * (1 + size_tol)

    total_span_width = 0.0
    total_span_chars = 0
    line_widths = []

    for page in doc:
        pd = page.get_text("dict")
        for b in pd.get("blocks", []):
            if b.get("type", 0) != 0:
                continue
            for ln in b.get("lines", []):
                # weighted size filter (body-like)
                lwsize = line_weighted_size(ln)
                if not (low <= lwsize <= high):
                    continue

                # collect per-line width
                x0s, x1s = [], []
                for sp in ln.get("spans", []):
                    txt = clean_text(sp.get("text", ""))
                    if not txt:
                        continue
                    x0, y0, x1, y1 = sp.get("bbox", (0, 0, 0, 0))
                    x0s.append(x0)
                    x1s.append(x1)
                    # accumulate average char width at body size
                    total_span_width += x1 - x0
                    total_span_chars += len(txt)

                if x0s and x1s:
                    line_widths.append(max(x1s) - min(x0s))

    if total_span_chars == 0 or not line_widths:
        return (None, None, None, None)

    avg_char_w = total_span_width / total_span_chars  # points / char
    typical_col_w = _quantile(line_widths, WIDTH_QUANTILE)  # points
    expected_capacity = int(
        round(typical_col_w / avg_char_w)
    )  # chars in a full-width body line
    dynamic_min = max(MIN_LEN_FLOOR, int(expected_capacity * MIN_OCCUPANCY_RATIO))
    return (dynamic_min, expected_capacity, avg_char_w, typical_col_w)


def extract_body_line_lengths(pdf_path: str):
    doc = fitz.open(pdf_path)
    body_size = weighted_mode_font_size(doc)
    dyn_min, capacity, avg_w, col_w = estimate_dynamic_min_len(
        doc, body_size, size_tol=SIZE_TOLERANCE
    )

    # Fallback if estimation fails:
    effective_min_len = dyn_min if dyn_min is not None else MIN_LINE_LEN

    if body_size <= 0:
        return {
            "pdf": pdf_path,
            "error": "No text detected (PDF may be scanned or empty).",
        }

    accepted_lengths = []
    sample_lines = []

    low, high = body_size * (1 - SIZE_TOLERANCE), body_size * (1 + SIZE_TOLERANCE)

    for page in doc:
        pd = page.get_text("dict")
        for b in pd.get("blocks", []):
            if b.get("type", 0) != 0:
                continue
            for ln in b.get("lines", []):
                # Build the raw line text from spans
                raw = " ".join(sp.get("text", "") for sp in ln.get("spans", []))
                txt = clean_text(raw)
                if not txt:
                    continue

                # quick skip: bullet-like starts (for length decision only)
                stripped_for_len = BULLET_PREFIX_RE.sub("", txt)
                # compute line's weighted size
                lwsize = line_weighted_size(ln)

                # keep if size around the body size and decent length
                if low <= lwsize <= high and len(stripped_for_len) >= effective_min_len:
                    accepted_lengths.append(
                        len(txt)
                    )  # count visible chars (incl. spaces)
                    # keep a few samples
                    if len(sample_lines) < 5:
                        sample_lines.append(txt)

    doc.close()

    if not accepted_lengths:
        return {
            "pdf": pdf_path,
            "body_font_size_pt": body_size,
            "warning": "No body-like lines found with current thresholds.",
            "recommendation": "Relax SIZE_TOLERANCE or lower MIN_LINE_LEN.",
        }

    # Stats
    med = statistics.median(accepted_lengths)
    mean = round(sum(accepted_lengths) / len(accepted_lengths), 2)
    mode_len = Counter(accepted_lengths).most_common(1)[0][0]

    return {
        "pdf": pdf_path,
        "body_font_size_pt": body_size,
        "lines_considered": len(accepted_lengths),
        "dynamic_threshold": {
            "effective_min_len": effective_min_len,
            "expected_full_line_capacity": capacity,
            "typical_column_width_pt": col_w,
            "avg_char_width_pt": avg_w,
            "occupancy_ratio": MIN_OCCUPANCY_RATIO,
            "width_quantile": WIDTH_QUANTILE,
        },
        "characters_per_line": {
            "median": med,
            "mean": mean,
            "mode": mode_len,
        },
        "examples": sample_lines,
    }


if __name__ == "__main__":
    import json

    file = "BaselFramework.pdf"
    result = extract_body_line_lengths(file)
    print(json.dumps(result, ensure_ascii=False, indent=2))
