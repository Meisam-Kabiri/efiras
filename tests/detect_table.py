from typing import Dict, List

import camelot
import fitz


# Try both methods
def any_camelot_latice_table_exist(file: str, page_num: int) -> bool:
    lattice_tables = camelot.read_pdf(file, pages=str(page_num), flavor="lattice")
    # stream_tables = camelot.read_pdf(file, pages=str(page_num), flavor='stream')

    # print(f"Lattice found: {len(lattice_tables)} tables")
    # print(f"Stream found: {len(stream_tables)} tables")

    if lattice_tables:
        return True
        print("Lattice result:")
        print(lattice_tables[0].df.head())

    return False


import re

import fitz


def extract_horizontal_lines_merged(
    pdf_path,
    page_num,
    acceptable_length=100,
    y_tolerance=2,
    x_gap_tolerance=5,
    skip_toc=True,
):
    """
    Your original function with simple TOC check
    """

    # Simple TOC check
    if skip_toc and is_table_of_contents(pdf_path, page_num):
        return {
            "lines": [],
            "skipped_toc": True,
            "message": f"Page {page_num} detected as TOC - skipped",
        }

    # Your original line extraction code
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]

    drawings = page.get_drawings()
    raw_lines = []

    # Step 1: Extract all potential horizontal lines
    for drawing in drawings:
        items = drawing.get("items", [])

        for item in items:
            if len(item) >= 3 and item[0] == "l":
                # Extract the points
                start_point = item[1]  # First Point
                end_point = item[2]  # Second Point

                width = abs(end_point[0] - start_point[0])
                height = abs(end_point[1] - start_point[1])

                if width > 10 and height < 5:
                    raw_lines.append(
                        {
                            "x_start": start_point[0],
                            "x_end": end_point[0],
                            "y_position": (end_point[1] + start_point[1]) / 2,
                            "thickness": height,
                            "type": "line",
                        }
                    )

    # Filter by acceptable length
    filtered_lines = [
        line
        for line in raw_lines
        if abs(line["x_end"] - line["x_start"]) >= acceptable_length
    ]

    doc.close()

    return {
        "lines": filtered_lines,
        "skipped_toc": False,
        "total_lines": len(filtered_lines),
    }


def extract_horizontal_lines_merged(
    pdf_path, page_num, acceptable_length=100, y_tolerance=2, x_gap_tolerance=5
):
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]  # Pymupdf starts from 0 instead of 1 (0-based index)

    drawings = page.get_drawings()
    raw_lines = []

    # Step 1: Extract all potential horizontal lines
    for drawing in drawings:
        items = drawing.get("items", [])

        # Method 1: Actual line commands with two points (start, end) instead of move-end
        # item is a list inside which is one or several tuple with several entries with following format
        # [('l', Point(194.68099975585938, 58.80804443359375), Point(427.12200927734375, 58.80804443359375))]
        for item in items:
            if len(item) >= 3 and item[0] == "l":
                # Extract the points
                start_point = item[1]  # First Point
                end_point = item[2]  # Second Point
                rect = item[1]
                width = abs(end_point[0] - start_point[0])
                height = abs(end_point[1] - start_point[1])

                if width > 10 and height < 5:
                    raw_lines.append(
                        {
                            "x_start": start_point[0],
                            "x_end": end_point[0],
                            "y_position": (end_point[1] + start_point[1]) / 2,
                            "thickness": height,
                            "type": "line",
                        }
                    )
                    # print('------------------------\n',raw_lines)

        # Method 1: Rectangle-based lines
        #  [('re', Rect(100.19999694824219, 242.97003173828125, 466.010009765625, 243.72003173828125), -1)]
        for item in items:
            if len(item) >= 2 and item[0] == "re":
                # Extract the points
                rect = item[1]  # Rectangle object (x0, y0, x1, y1)
                # Rectangle coordinates
                x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                width = abs(x1 - x0)
                height = abs(y1 - y0)

                # Check if it's a horizontal line-like rectangle
                if width > 10 and height < 5:
                    raw_lines.append(
                        {
                            "x_start": x0,
                            "x_end": x1,
                            "y_position": (y0 + y1) / 2,
                            "thickness": height,
                            "type": "rectangle",
                        }
                    )

        current_pos = None
        for item in items:
            if len(item) >= 2:
                if item[0] == "m":
                    current_pos = item[1]
                elif item[0] == "l" and current_pos:
                    end_pos = item[1]

                    # Check if horizontal
                    if (
                        abs(current_pos.y - end_pos.y) < y_tolerance
                        and abs(current_pos.x - end_pos.x) > 10
                    ):

                        raw_lines.append(
                            {
                                "x_start": min(current_pos.x, end_pos.x),
                                "x_end": max(current_pos.x, end_pos.x),
                                "y_position": (current_pos.y + end_pos.y) / 2,
                                "thickness": drawing.get("width", 1),
                                "type": "line_command",
                            }
                        )

                    current_pos = end_pos

        # Method 3: Actual line commands move-end
        current_pos = None
        for item in items:
            if len(item) >= 2:
                if item[0] == "m":
                    current_pos = item[1]
                elif item[0] == "l" and current_pos:
                    end_pos = item[1]

                    # Check if horizontal
                    if (
                        abs(current_pos.y - end_pos.y) < y_tolerance
                        and abs(current_pos.x - end_pos.x) > 10
                    ):

                        raw_lines.append(
                            {
                                "x_start": min(current_pos.x, end_pos.x),
                                "x_end": max(current_pos.x, end_pos.x),
                                "y_position": (current_pos.y + end_pos.y) / 2,
                                "thickness": drawing.get("width", 1),
                                "type": "move-end command",
                            }
                        )

                    current_pos = end_pos

    # Step 2: Merge lines with same Y position
    merged_lines = merge_horizontal_lines(raw_lines, y_tolerance, x_gap_tolerance)

    for lines in merged_lines:
        print(lines)
        print("------------------------\n")

    doc.close()
    return merged_lines


def merge_horizontal_lines(lines, y_tolerance=2, x_gap_tolerance=5):
    """Merge horizontal lines that are on the same Y level"""
    if not lines:
        return []

    # Sort by Y position first, then by X start
    lines.sort(key=lambda x: (x["y_position"], x["x_start"]))

    merged = []
    current_group = [lines[0]]

    for line in lines[1:]:
        last_line = current_group[-1]

        # Check if lines are on same Y level
        y_diff = abs(line["y_position"] - last_line["y_position"])

        if y_diff <= y_tolerance:
            # Same Y level - check if they should be merged
            current_group.append(line)
        else:
            # Different Y level - process current group and start new one
            merged.extend(merge_line_group(current_group, x_gap_tolerance))
            current_group = [line]

    # Process the last group
    merged.extend(merge_line_group(current_group, x_gap_tolerance))

    return merged


def merge_line_group(line_group, x_gap_tolerance=5):
    """Merge lines in the same horizontal group"""
    if len(line_group) == 1:
        return line_group

    # Sort by x_start
    line_group.sort(key=lambda x: x["x_start"])

    merged_lines = []
    current_line = line_group[0].copy()

    for next_line in line_group[1:]:
        # Check if lines are close enough to merge
        gap = next_line["x_start"] - current_line["x_end"]

        if gap <= x_gap_tolerance:
            # Merge: extend current line to include next line
            current_line["x_end"] = max(current_line["x_end"], next_line["x_end"])
            current_line["thickness"] = max(
                current_line["thickness"], next_line["thickness"]
            )

            # Update merged info
            if "merged_count" not in current_line:
                current_line["merged_count"] = 2
                current_line["merged_types"] = [current_line["type"], next_line["type"]]
            else:
                current_line["merged_count"] += 1
                current_line["merged_types"].append(next_line["type"])

        else:
            # Gap too big - save current line and start new one
            merged_lines.append(current_line)
            current_line = next_line.copy()

    # Add the last line
    merged_lines.append(current_line)

    return merged_lines


# # Usage with debugging
# def analyze_line_merging(pdf_path, page_num=0):
#     doc = fitz.open(pdf_path)
#     page = doc[page_num]

#     print("=== Before Merging ===")
#     raw_lines = extract_horizontal_lines_merged(pdf_path, page_num)

#     print(f"Found {len(raw_lines)} total horizontal lines")

#     for i, line in enumerate(raw_lines):
#         print(f"Line {i}: Y={line['y_position']:.1f}, "
#               f"X={line['x_start']:.1f} to {line['x_end']:.1f}, "
#               f"width={line['x_end']-line['x_start']:.1f}")

#         if 'merged_count' in line:
#             print(f"  → Merged from {line['merged_count']} segments: {line['merged_types']}")

#     doc.close()
#     return raw_lines


def if_contain_horiontal_line(file: str, page_num: int) -> bool:
    doc = fitz.open(file)
    page = doc[page_num]
    rect = page.rect
    width = rect.width
    height = rect.height
    print(rect.y0)
    print(rect.y1)

    print(f"Page {page_num + 1}: {width:.0f} x {height:.0f} points")
    # Convert to other units if needed
    print(f"Page {page_num + 1}: {width/72:.2f} x {height/72:.2f} inches")

    margin_threshold = 0.03
    acceptable_horizontal_line_margin = [
        rect.y0 + margin_threshold * height,
        rect.y1 - margin_threshold * height,
    ]
    print(acceptable_horizontal_line_margin)

    doc.close()


def merge_single_digits(lines):
    merged = []
    i = 0
    while i < len(lines):
        current_line = lines[i]

        # Check if next line is just a digit
        if (
            i + 1 < len(lines)
            and lines[i + 1].strip().isdigit()
            and len(lines[i + 1].strip()) <= 3
        ):

            # Merge current line with the digit
            merged_line = current_line + " " + lines[i + 1].strip()
            merged.append(merged_line)
            i += 2  # Skip both lines
        else:
            merged.append(current_line)
            i += 1

    return merged


def is_table_of_contents(pdf_path, page_num):
    """
    Simple TOC detection based on pattern: text + dots/spaces + number

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-based)
        min_blocks: Minimum number of text blocks to analyze
        allow_exceptions: Number of blocks that can NOT match the pattern

    Returns:
        bool: True if page is likely a TOC
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]

    # Get text blocks (better than splitting by \n)
    blocks = page.get_text("blocks")
    text_blocks = []

    # Extract text content from blocks
    for block in blocks:
        if len(block) >= 5 and block[4].strip():  # block[4] contains the text
            text_blocks.append(block[4].strip())

    doc.close()

    # TOC pattern: text + (dots or spaces) + number at end
    # Matches patterns like:
    # "Chapter 1 Introduction ........ 5"
    # "Section 2.1 Methods      15"
    # "Appendix A             123"
    toc_pattern = re.compile(r".+[\.\s]{2,}\s*\d+\s*$")

    matching_blocks = 0
    num_lines = 0
    for block in text_blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        merged_lines = merge_single_digits(lines)
        num_lines += len(lines)

        for line in lines:

            if toc_pattern.match(line) and len(line) > 15:
                matching_blocks += 1

        for line in merged_lines:

            if toc_pattern.match(line) and len(line) > 15:
                matching_blocks += 1

    # Allow some exceptions (non-matching blocks)
    exceptions = len(text_blocks) - matching_blocks

    return True if matching_blocks >= 2 else False


file = "BaselFramework.pdf"
file = "CV.pdf"
file = "Meisam_thesis.pdf"

is_toc = is_table_of_contents(file, 3)
print(is_toc)
for i in range(0, 193):
    is_toc = is_table_of_contents(file, i)
    if is_toc:
        print(f"Page {i} is TOC")


# b = any_camelot_latice_table_exist(file, 0)
# print(b)

# if_contain_horiontal_line(file,0)

# lines = extract_horizontal_lines_merged(file, 20)
