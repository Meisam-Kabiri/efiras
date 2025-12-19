# pip install pymupdf azure-ai-documentintelligence
import os

import fitz
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

PDF_PATH = "BaselFramework.pdf"
PDF_PATH = "Meisam_thesis.pdf"

PAGE = 172  # <-- 1-based page you want
FREE_TIER_LIMIT = 4 * 1024 * 1024  # 4 MB

# --- slice just PAGE=20 into a 1-page PDF ---
src = fitz.open(PDF_PATH)
assert 1 <= PAGE <= len(src), f"PAGE out of range (1..{len(src)})"
one = fitz.open()
one.insert_pdf(src, from_page=PAGE - 1, to_page=PAGE - 1)

# compact bytes (no linearization)
pdf_bytes = one.tobytes(deflate=True, garbage=4, clean=True)
print(f"Single-page PDF size: {len(pdf_bytes)/1024:.1f} KB")

# if still too big for F0, render to JPEG
body = pdf_bytes
if len(pdf_bytes) > FREE_TIER_LIMIT:
    print("PDF >4MB on F0 → falling back to JPEG…")
    pix = one[0].get_pixmap(matrix=fitz.Matrix(180 / 72, 180 / 72), alpha=False)
    body = pix.tobytes("jpeg", jpg_quality=75)
    print(f"JPEG size: {len(body)/1024:.1f} KB")

client = DocumentIntelligenceClient(
    endpoint=os.environ["DOCUMENTINTELLIGENCE_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["DOCUMENTINTELLIGENCE_API_KEY"]),
)

try:
    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=body,
        pages="1",  # IMPORTANT: we uploaded a 1-page doc/image
    )
    result = poller.result()
except HttpResponseError as e:
    print("Azure error:", e)
    raise

tables = result.tables or []
print(f"Tables found on page 20: {len(tables)}")

if tables:
    for t in tables:
        grid = {r: {} for r in range(t.row_count)}
        for c in t.cells:
            grid[c.row_index][c.column_index] = c.content or ""
        print(f"\nTable ({t.row_count}x{t.column_count})")
        for r in range(t.row_count):
            print(",".join(grid[r].get(k, "") for k in range(t.column_count)))
else:
    preview = getattr(result, "content", "") or ""
    if not preview and getattr(result, "paragraphs", None):
        preview = " ".join(
            p.content for p in result.paragraphs if getattr(p, "content", None)
        )
    print("No tables detected. Text preview (first 300 chars):")
    print((preview or "[no text extracted]")[:300])


import csv
import re

import pandas as pd

all_rows = []
for t in result.tables or []:
    # convert Azure table -> rows
    rows = [
        [
            next(
                (
                    c.content or ""
                    for c in t.cells
                    if c.row_index == r and c.column_index == k
                ),
                "",
            )
            for k in range(t.column_count)
        ]
        for r in range(t.row_count)
    ]

    # forward-fill first column (Category) where Azure left it blank
    last = ""
    for r in rows:
        if r and r[0].strip() == "":
            r[0] = last
        else:
            last = r[0]

    # clean: drop trailing '*' from indicator, strip '%' from weighting
    for r in rows[1:]:
        if len(r) >= 2:
            r[1] = re.sub(r"\*$", "", r[1]).strip()
        if len(r) >= 3:
            r[2] = r[2].replace("%", "").strip()

    all_rows.extend(rows)

# Save CSV
with open("page20_table1.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(all_rows)

# Or make a DataFrame (rename cols if you like)
df = pd.DataFrame(all_rows[1:], columns=all_rows[0])
print(df)
df.to_csv("page20_table1_clean.csv", index=False)
