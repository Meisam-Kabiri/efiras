# pip install azure-ai-documentintelligence
import os

import fitz
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

# prebuilt-read and prebuilt-layout models are available and working in your Azure Document Intelligence resource
# prebuilt-document is not available (likely due to service tier or regional limitations)
# The alternative model names (read, layout, document) are also not recognized


# Load credentials from environment
endpoint = os.environ["DOCUMENTINTELLIGENCE_ENDPOINT"]
api_key = os.environ["DOCUMENTINTELLIGENCE_API_KEY"]

client = DocumentIntelligenceClient(
    endpoint=endpoint, credential=AzureKeyCredential(api_key)
)


def analyze_page_sequential_blocks(pdf_path: str, page_number: int):
    """
    Analyze a PDF page using Azure Document Intelligence and create sequential document blocks.

    AZURE DOCUMENT INTELLIGENCE MODEL OPTIONS:
    ==========================================

    1. 🔤 PREBUILT-READ:
       - Basic text extraction only
       - Extracts text line by line (OCR)
       - NO table structure detection
       - NO key-value pairs
       - ✅ Available on FREE TIER
       - Best for: Simple text extraction from documents

    2. 📊 PREBUILT-LAYOUT:
       - Advanced text + layout analysis
       - Extracts text AND identifies tables, forms, selection marks
       - Detects table structure (rows/columns)
       - Preserves reading order and spatial relationships
       - ✅ Available on FREE TIER
       - Best for: Documents with tables, forms, complex layouts
       - ⚠️  QUIRK: Extracts text sequentially first (including table content as messy text),
                  then separately extracts structured tables. This function creates clean blocks.

    3. 📋 PREBUILT-DOCUMENT:
       - Most comprehensive analysis
       - Everything from prebuilt-layout PLUS key-value pairs, entities
       - Advanced document understanding (forms, invoices, receipts)
       - ❌ NOT AVAILABLE on FREE/STUDENT TIER (requires paid subscription)
       - Best for: Business documents, forms processing, invoice analysis

    CURRENT FUNCTION BEHAVIOR:
    =========================
    - Uses PREBUILT-LAYOUT model (best available on free tier)
    - Creates sequential document blocks in proper reading order
    - Each block is either TEXT or TABLE (no overlap, no repetition)
    - Maintains document flow: Table Title → Table → Analysis Text → etc.
    """
    FREE_TIER_LIMIT = 4 * 1024 * 1024  # 4 MB

    # --- slice just the specified page into a 1-page PDF ---
    src = fitz.open(pdf_path)
    assert 1 <= page_number <= len(src), f"PAGE out of range (1..{len(src)})"
    one = fitz.open()
    one.insert_pdf(src, from_page=page_number - 1, to_page=page_number - 1)

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

    # Clean up
    one.close()
    src.close()

    # Use prebuilt-layout for comprehensive extraction
    poller = client.begin_analyze_document(model_id="prebuilt-layout", body=body)
    result = poller.result()

    return create_sequential_blocks_from_sections(result)


def create_sequential_blocks_from_sections(result):
    blocks = []

    if not result.sections:
        return blocks

    # Check if sections exist and aren't empty
    if hasattr(result, "sections") and result.sections:

        print(f"Found {len(result.sections)} sections")
        section = result.sections[0]  # Main document section

        # The elements are already in reading order!
        for element_ref in section["elements"]:

            if element_ref.startswith("/paragraphs/"):
                # Extract index: '/paragraphs/5' -> 5
                para_index = int(element_ref.split("/")[-1])
                paragraph = result.paragraphs[para_index]

                blocks.append({"type": "TEXT", "content": paragraph.content})

            elif element_ref.startswith("/tables/"):
                # Extract index: '/tables/0' -> 0
                table_index = int(element_ref.split("/")[-1])
                table = result.tables[table_index]

                blocks.append({"type": "TABLE", "content": format_table(table)})
    else:
        print("No valid sections found.")
        table_y_regions = []

        for tab in result.tables:
            polygon = tab["boundingRegions"][0]["polygon"]  # x1,y1,x2,y2,x3,y3,x4,y4
            y_coords = [polygon[1], polygon[3], polygon[5], polygon[7]]
            table_y_regions.append([min(y_coords), max(y_coords)])
            blocks.append(
                {"type": "TABLE", "content": format_table(tab), "top_side": polygon[1]}
            )

        for par in result.paragraphs:
            polygon = par["boundingRegions"][0]["polygon"]  # x1,y1,x2,y2,x3,y3,x4,y4

            is_in_table = False
            for y_min, y_max in table_y_regions:
                # Skip paragraphs that overlap with any table region
                if y_min <= polygon[1] <= y_max:
                    is_in_table = True
                    break
            if not is_in_table:
                blocks.append(
                    {"type": "TEXT", "content": par.content, "top_side": polygon[1]}
                )

    blocks.sort(key=lambda x: x.get("top_side", 0))
    return blocks


def get_line_y_position(line):
    """Extract Y coordinate from line for positioning"""
    if hasattr(line, "bounding_box") and line.bounding_box:
        return line.bounding_box[1]  # Y coordinate of top-left
    return 0


def format_table(table):
    """Format table into a clean structure"""
    rows = {}
    max_col = 0

    for cell in table.cells:
        row_idx = cell.row_index
        col_idx = cell.column_index
        max_col = max(max_col, col_idx)

        if row_idx not in rows:
            rows[row_idx] = {}
        rows[row_idx][col_idx] = cell.content.strip()

    # Convert to list format
    formatted_rows = []
    headers = None

    for row_idx in sorted(rows.keys()):
        row_data = []
        for col_idx in range(max_col + 1):
            cell_content = rows[row_idx].get(col_idx, "")
            row_data.append(cell_content)

        if row_idx == 0:
            headers = row_data
        formatted_rows.append(row_data)

    return {
        "headers": headers,
        "rows": formatted_rows,
        "row_count": len(formatted_rows),
        "col_count": max_col + 1,
    }


def print_sequential_blocks(blocks):
    """
    Print document blocks in clean, sequential order
    """
    print("\n" + "=" * 60)
    print("📄 SEQUENTIAL DOCUMENT BLOCKS")
    print("=" * 60)

    for i, block in enumerate(blocks, 1):
        if block["type"] == "CAPTION":
            print(f"\n📝 Block {i} - TABLE CAPTION:")
            print("─" * 40)
            print(f"   {block['content']}")

        elif block["type"] == "TABLE":
            print(f"\n📊 Block {i} - TABLE:")
            print("─" * 40)
            table_data = block["content"]

            if table_data["headers"]:
                headers = table_data["headers"]
                col_widths = [max(len(str(cell)), 8) for cell in headers]

                # Adjust column widths based on data
                for row in table_data["rows"][1:]:  # Skip header row
                    for j, cell in enumerate(row):
                        if j < len(col_widths):
                            col_widths[j] = max(col_widths[j], len(str(cell)))

                # Print table with proper alignment
                header_row = " | ".join(
                    f"{h:<{col_widths[j]}}" for j, h in enumerate(headers)
                )
                print(f"   {header_row}")
                print(f"   {'-' * len(header_row)}")

                for row in table_data["rows"][1:]:  # Skip header
                    formatted_row = " | ".join(
                        f"{row[j] if j < len(row) else '':<{col_widths[j]}}"
                        for j in range(len(col_widths))
                    )
                    print(f"   {formatted_row}")

        elif block["type"] == "TEXT":
            print(f"\n📖 Block {i} - TEXT:")
            print("─" * 40)
            # Format long text with proper line breaks
            content = block["content"]
            if len(content) > 80:
                # Break long paragraphs into readable chunks
                words = content.split()
                lines = []
                current_line = []
                current_length = 0

                for word in words:
                    if current_length + len(word) + 1 > 80:
                        lines.append(" ".join(current_line))
                        current_line = [word]
                        current_length = len(word)
                    else:
                        current_line.append(word)
                        current_length += len(word) + 1

                if current_line:
                    lines.append(" ".join(current_line))

                for line in lines:
                    print(f"   {line}")
            else:
                print(f"   {content}")


# Main execution with improved documentation
if __name__ == "__main__":
    pdf_file = "Meisam_thesis.pdf"
    page_num = 114

    print("AZURE DOCUMENT INTELLIGENCE - SEQUENTIAL BLOCK ANALYSIS")
    print("=" * 60)
    print("🎯 Model: prebuilt-layout (FREE TIER)")
    print("🔧 Strategy: Sequential blocks without overlap")
    print("📋 Block types: CAPTION → TABLE → TEXT (in reading order)")
    print("=" * 60)

    # Create sequential blocks
    blocks = analyze_page_sequential_blocks(pdf_file, page_num)
    print_sequential_blocks(blocks)

    # pdf_file = "Meisam_thesis.pdf"
    # page_num = 114
    # outputs = analyze_page_all_models(pdf_file, page_num)

    # for model, content in outputs.items():
    #     print(f"\n=== {model.upper()} ===")
    #     print("\n-- Text --")
    #     for line in content["text"]:
    #         print(line)

    #     if content["tables"]:
    #         print("\n-- Tables --")
    #         for row in content["tables"]:
    #             print(row)

    #     if content["key_value_pairs"]:
    #         print("\n-- Key-Value Pairs --")
    #         for kv in content["key_value_pairs"]:
    #             print(kv)
