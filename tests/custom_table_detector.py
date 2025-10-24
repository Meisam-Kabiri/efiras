#!/usr/bin/env python3
"""
Custom Table Detection System

Uses multiple detection methods to precisely locate tables, then uses Camelot
for extraction. Avoids false positives by being more intelligent about what
constitutes a real table.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np

try:
    import camelot

    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    print("⚠️  Camelot not available for extraction")


class CustomTableDetector:
    """Intelligent table detection using multiple heuristics."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.filename = Path(pdf_path).name

    def detect_tables_all_methods(self) -> Dict[str, Any]:
        """Run all detection methods and combine results."""
        print(f"\n🔍 Custom Table Detection: {self.filename}")
        print("=" * 60)

        all_detections = {
            "filename": self.filename,
            "total_pages": len(self.doc),
            "detection_methods": {},
            "combined_tables": [],
            "summary": {},
        }

        # Run each detection method
        methods = [
            ("keyword_detection", self.detect_by_keywords),
            ("line_pattern_detection", self.detect_by_line_patterns),
            ("text_alignment_detection", self.detect_by_text_alignment),
            ("visual_cues_detection", self.detect_by_visual_cues),
        ]

        for method_name, method_func in methods:
            print(f"\n🎯 Running {method_name.replace('_', ' ').title()}...")
            try:
                results = method_func()
                all_detections["detection_methods"][method_name] = results
                print(f"   Found {len(results)} potential table areas")
            except Exception as e:
                print(f"   ❌ {method_name} failed: {e}")
                all_detections["detection_methods"][method_name] = []

        # Combine and deduplicate results
        combined_tables = self.combine_detections(all_detections["detection_methods"])
        all_detections["combined_tables"] = combined_tables

        # Extract tables using Camelot if available
        if CAMELOT_AVAILABLE and combined_tables:
            extracted_tables = self.extract_tables_with_camelot(combined_tables)
            all_detections["extracted_tables"] = extracted_tables

        # Generate summary
        all_detections["summary"] = self.generate_detection_summary(all_detections)

        return all_detections

    def detect_by_keywords(self) -> List[Dict[str, Any]]:
        """Detect tables by looking for table captions and labels."""
        table_areas = []

        # Patterns that indicate tables
        table_patterns = [
            r"\btable\s+\d+\.?\d*\b",  # "Table 1", "Table 2.1"
            r"\btab\s+\d+\.?\d*\b",  # "Tab 1"
            r"table\s*:",  # "Table:"
            r"tab\s*:",  # "Tab:"
            r"tabelle\s+\d+",  # German: "Tabelle 1"
            r"tableau\s+\d+",  # French: "Tableau 1"
        ]

        compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in table_patterns
        ]

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            text_dict = page.get_text("dict")

            # Find table keywords
            for pattern in compiled_patterns:
                matches = pattern.finditer(text)

                for match in matches:
                    keyword = match.group()
                    start_pos = match.start()

                    # Find the text block containing this keyword
                    keyword_block = self.find_text_block_by_position(
                        text_dict, keyword, start_pos
                    )

                    if keyword_block:
                        # Look for table content below the keyword
                        table_area = self.find_table_area_near_keyword(
                            page, keyword_block, page_num
                        )

                        if table_area:
                            table_areas.append(
                                {
                                    "page": page_num + 1,
                                    "method": "keyword_detection",
                                    "keyword": keyword,
                                    "bbox": table_area,
                                    "confidence": 0.8,
                                    "keyword_block": keyword_block,
                                }
                            )

        return table_areas

    def detect_by_line_patterns(self) -> List[Dict[str, Any]]:
        """Detect tables by analyzing line patterns (grids)."""
        table_areas = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            drawings = page.get_drawings()

            # Collect all lines
            horizontal_lines = []
            vertical_lines = []

            for drawing in drawings:
                items = drawing.get("items", [])
                for item in items:
                    if "l" in item:  # Line item
                        line_data = item["l"]
                        # Extract line coordinates
                        if len(line_data) >= 4:
                            x1, y1, x2, y2 = line_data[:4]

                            # Classify as horizontal or vertical
                            if abs(y1 - y2) < 2:  # Horizontal line
                                horizontal_lines.append((x1, y1, x2, y2))
                            elif abs(x1 - x2) < 2:  # Vertical line
                                vertical_lines.append((x1, y1, x2, y2))

            # Find intersecting grid patterns
            grids = self.find_grid_patterns(horizontal_lines, vertical_lines)

            for grid in grids:
                table_areas.append(
                    {
                        "page": page_num + 1,
                        "method": "line_pattern_detection",
                        "bbox": grid["bbox"],
                        "grid_cells": grid["cells"],
                        "horizontal_lines": len(grid["h_lines"]),
                        "vertical_lines": len(grid["v_lines"]),
                        "confidence": grid["confidence"],
                    }
                )

        return table_areas

    def detect_by_text_alignment(self) -> List[Dict[str, Any]]:
        """Detect tables by finding consistently aligned text blocks."""
        table_areas = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text_dict = page.get_text("dict")

            # Analyze text block alignment
            blocks = [b for b in text_dict["blocks"] if "lines" in b]

            if len(blocks) < 4:  # Need multiple blocks for alignment analysis
                continue

            # Group blocks by Y-coordinate (potential rows)
            row_groups = self.group_blocks_by_rows(blocks)

            # Find groups with multiple aligned columns
            table_candidates = []
            for y_coord, row_blocks in row_groups.items():
                if len(row_blocks) >= 2:  # At least 2 columns
                    # Check if blocks are evenly spaced (table-like)
                    if self.blocks_are_table_like(row_blocks):
                        table_candidates.append(row_blocks)

            # If we have multiple aligned rows, it's likely a table
            if len(table_candidates) >= 2:
                all_table_blocks = [block for row in table_candidates for block in row]
                bbox = self.calculate_blocks_bbox(all_table_blocks)

                # Check content quality
                confidence = self.calculate_table_confidence(table_candidates)

                if confidence > 0.6:  # Only high-confidence detections
                    table_areas.append(
                        {
                            "page": page_num + 1,
                            "method": "text_alignment_detection",
                            "bbox": bbox,
                            "rows": len(table_candidates),
                            "avg_cols": sum(len(row) for row in table_candidates)
                            / len(table_candidates),
                            "confidence": confidence,
                            "blocks": len(all_table_blocks),
                        }
                    )

        return table_areas

    def detect_by_visual_cues(self) -> List[Dict[str, Any]]:
        """Detect tables by visual cues like borders, spacing, etc."""
        table_areas = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]

            # Look for rectangular regions that might be tables
            rectangles = self.find_table_rectangles(page)

            # Look for areas with regular spacing patterns
            spacing_patterns = self.find_regular_spacing_patterns(page)

            # Combine visual cues
            visual_areas = rectangles + spacing_patterns

            for area in visual_areas:
                # Validate that area contains table-like content
                if self.validate_table_content(page, area["bbox"]):
                    table_areas.append(
                        {
                            "page": page_num + 1,
                            "method": "visual_cues_detection",
                            "bbox": area["bbox"],
                            "cue_type": area["type"],
                            "confidence": area["confidence"],
                        }
                    )

        return table_areas

    def find_text_block_by_position(
        self, text_dict: Dict, keyword: str, pos: int
    ) -> Optional[Dict]:
        """Find the text block containing a specific keyword."""
        current_pos = 0

        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        span_text = span["text"]
                        if current_pos <= pos < current_pos + len(span_text):
                            if keyword.lower() in span_text.lower():
                                return block
                        current_pos += len(span_text)

        return None

    def find_table_area_near_keyword(
        self, page, keyword_block: Dict, page_num: int
    ) -> Optional[List[float]]:
        """Find table content area near a table keyword."""
        keyword_bbox = keyword_block["bbox"]
        page_height = page.rect.height

        # Look for content below the keyword (typical table position)
        search_area = [
            keyword_bbox[0] - 50,  # Extend left a bit
            keyword_bbox[3],  # Start below keyword
            keyword_bbox[2] + 50,  # Extend right a bit
            min(keyword_bbox[3] + 200, page_height),  # Look down 200pt or page end
        ]

        # Check if this area contains table-like content
        if self.area_contains_table_content(page, search_area):
            return search_area

        return None

    def area_contains_table_content(self, page, bbox: List[float]) -> bool:
        """Check if an area contains table-like content."""
        # Extract text from the area
        rect = fitz.Rect(bbox)
        text = page.get_text("text", clip=rect)

        if not text.strip():
            return False

        lines = text.strip().split("\n")

        # Heuristics for table content:
        # 1. Multiple lines
        if len(lines) < 2:
            return False

        # 2. Contains numbers or structured data
        number_pattern = re.compile(r"\d+(?:\.\d+)?")
        lines_with_numbers = sum(1 for line in lines if number_pattern.search(line))

        if lines_with_numbers / len(lines) > 0.3:  # 30% of lines have numbers
            return True

        # 3. Consistent formatting (similar line lengths)
        line_lengths = [len(line.strip()) for line in lines if line.strip()]
        if line_lengths:
            avg_length = sum(line_lengths) / len(line_lengths)
            similar_lengths = sum(
                1 for l in line_lengths if abs(l - avg_length) < avg_length * 0.3
            )

            if similar_lengths / len(line_lengths) > 0.6:  # 60% similar lengths
                return True

        return False

    def group_blocks_by_rows(self, blocks: List[Dict]) -> Dict[float, List[Dict]]:
        """Group text blocks by Y-coordinate (rows)."""
        row_groups = defaultdict(list)
        tolerance = 5  # pixels

        for block in blocks:
            bbox = block["bbox"]
            y_center = (bbox[1] + bbox[3]) / 2

            # Find existing group or create new one
            found_group = False
            for existing_y in row_groups:
                if abs(y_center - existing_y) < tolerance:
                    row_groups[existing_y].append(block)
                    found_group = True
                    break

            if not found_group:
                row_groups[y_center] = [block]

        return row_groups

    def blocks_are_table_like(self, blocks: List[Dict]) -> bool:
        """Check if blocks are arranged like table cells."""
        if len(blocks) < 2:
            return False

        # Sort blocks by X coordinate
        blocks_sorted = sorted(blocks, key=lambda b: b["bbox"][0])

        # Check if blocks are evenly spaced
        x_positions = [b["bbox"][0] for b in blocks_sorted]

        if len(x_positions) >= 3:
            # Check for regular spacing
            gaps = [
                x_positions[i + 1] - x_positions[i] for i in range(len(x_positions) - 1)
            ]
            avg_gap = sum(gaps) / len(gaps)
            regular_gaps = sum(1 for gap in gaps if abs(gap - avg_gap) < avg_gap * 0.3)

            return regular_gaps / len(gaps) > 0.7  # 70% regular spacing

        return True  # 2 blocks are fine

    def calculate_blocks_bbox(self, blocks: List[Dict]) -> List[float]:
        """Calculate bounding box for a list of blocks."""
        if not blocks:
            return [0, 0, 0, 0]

        min_x = min(b["bbox"][0] for b in blocks)
        min_y = min(b["bbox"][1] for b in blocks)
        max_x = max(b["bbox"][2] for b in blocks)
        max_y = max(b["bbox"][3] for b in blocks)

        return [min_x, min_y, max_x, max_y]

    def calculate_table_confidence(self, table_candidates: List[List[Dict]]) -> float:
        """Calculate confidence score for detected table."""
        if not table_candidates:
            return 0.0

        confidence = 0.5  # Base confidence

        # More rows = higher confidence
        row_count = len(table_candidates)
        confidence += min(row_count * 0.1, 0.3)

        # Consistent column count = higher confidence
        col_counts = [len(row) for row in table_candidates]
        if len(set(col_counts)) == 1:  # All rows have same column count
            confidence += 0.2

        # Check for numeric content
        all_text = ""
        for row in table_candidates:
            for block in row:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            all_text += span["text"] + " "

        if re.search(r"\d+", all_text):  # Contains numbers
            confidence += 0.1

        return min(confidence, 1.0)

    def find_grid_patterns(
        self, h_lines: List[Tuple], v_lines: List[Tuple]
    ) -> List[Dict]:
        """Find grid patterns from horizontal and vertical lines."""
        grids = []

        if len(h_lines) < 2 or len(v_lines) < 2:
            return grids

        # Group nearby parallel lines
        h_groups = self.group_parallel_lines(h_lines, is_horizontal=True)
        v_groups = self.group_parallel_lines(v_lines, is_horizontal=False)

        # Find intersecting groups that form grids
        for h_group in h_groups:
            for v_group in v_groups:
                if self.lines_intersect_as_grid(h_group, v_group):
                    grid_bbox = self.calculate_grid_bbox(h_group, v_group)

                    grids.append(
                        {
                            "bbox": grid_bbox,
                            "h_lines": h_group,
                            "v_lines": v_group,
                            "cells": (len(h_group) - 1) * (len(v_group) - 1),
                            "confidence": min(len(h_group) * len(v_group) / 10, 1.0),
                        }
                    )

        return grids

    def group_parallel_lines(
        self, lines: List[Tuple], is_horizontal: bool
    ) -> List[List[Tuple]]:
        """Group parallel lines that might form table borders."""
        if not lines:
            return []

        groups = []
        tolerance = 20  # pixels

        for line in lines:
            added_to_group = False

            for group in groups:
                # Check if line is parallel and nearby to group
                if self.line_fits_in_group(line, group, is_horizontal, tolerance):
                    group.append(line)
                    added_to_group = True
                    break

            if not added_to_group:
                groups.append([line])

        # Filter groups with at least 2 lines
        return [group for group in groups if len(group) >= 2]

    def line_fits_in_group(
        self, line: Tuple, group: List[Tuple], is_horizontal: bool, tolerance: float
    ) -> bool:
        """Check if a line fits in a group of parallel lines."""
        if not group:
            return True

        x1, y1, x2, y2 = line

        for existing_line in group:
            ex1, ey1, ex2, ey2 = existing_line

            if is_horizontal:
                # Check Y coordinate similarity
                if abs(y1 - ey1) < tolerance and abs(y2 - ey2) < tolerance:
                    return True
            else:
                # Check X coordinate similarity
                if abs(x1 - ex1) < tolerance and abs(x2 - ex2) < tolerance:
                    return True

        return False

    def lines_intersect_as_grid(
        self, h_lines: List[Tuple], v_lines: List[Tuple]
    ) -> bool:
        """Check if horizontal and vertical lines intersect to form a grid."""
        # Simplified check - at least some intersections
        intersections = 0

        for h_line in h_lines:
            hx1, hy1, hx2, hy2 = h_line
            for v_line in v_lines:
                vx1, vy1, vx2, vy2 = v_line

                # Check if lines intersect
                if (
                    min(hx1, hx2) <= max(vx1, vx2)
                    and max(hx1, hx2) >= min(vx1, vx2)
                    and min(vy1, vy2) <= max(hy1, hy2)
                    and max(vy1, vy2) >= min(hy1, hy2)
                ):
                    intersections += 1

        # Need at least 4 intersections for a minimal grid
        return intersections >= 4

    def calculate_grid_bbox(
        self, h_lines: List[Tuple], v_lines: List[Tuple]
    ) -> List[float]:
        """Calculate bounding box for a grid."""
        all_x = [x for line in h_lines + v_lines for x in [line[0], line[2]]]
        all_y = [y for line in h_lines + v_lines for y in [line[1], line[3]]]

        return [min(all_x), min(all_y), max(all_x), max(all_y)]

    def find_table_rectangles(self, page) -> List[Dict]:
        """Find rectangular shapes that might be table borders."""
        rectangles = []
        drawings = page.get_drawings()

        for drawing in drawings:
            items = drawing.get("items", [])
            for item in items:
                if "re" in item:  # Rectangle item
                    rect_data = item["re"]
                    if len(rect_data) >= 4:
                        x, y, w, h = rect_data[:4]

                        # Filter by size (tables are usually reasonably sized)
                        if w > 100 and h > 50:  # Minimum table size
                            rectangles.append(
                                {
                                    "type": "rectangle_border",
                                    "bbox": [x, y, x + w, y + h],
                                    "confidence": 0.7,
                                }
                            )

        return rectangles

    def find_regular_spacing_patterns(self, page) -> List[Dict]:
        """Find areas with regular spacing that might indicate tables."""
        patterns = []

        text_dict = page.get_text("dict")
        blocks = [b for b in text_dict["blocks"] if "lines" in b]

        # Look for blocks with very regular Y spacing
        if len(blocks) >= 4:
            y_positions = sorted([b["bbox"][1] for b in blocks])
            gaps = [
                y_positions[i + 1] - y_positions[i] for i in range(len(y_positions) - 1)
            ]

            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                regular_gaps = sum(
                    1 for gap in gaps if abs(gap - avg_gap) < 5
                )  # 5pt tolerance

                if regular_gaps / len(gaps) > 0.8:  # 80% regular spacing
                    bbox = self.calculate_blocks_bbox(blocks)
                    patterns.append(
                        {"type": "regular_spacing", "bbox": bbox, "confidence": 0.6}
                    )

        return patterns

    def validate_table_content(self, page, bbox: List[float]) -> bool:
        """Validate that a visual area contains table-like content."""
        return self.area_contains_table_content(page, bbox)

    def combine_detections(self, method_results: Dict[str, List[Dict]]) -> List[Dict]:
        """Combine results from different detection methods, removing duplicates."""
        all_detections = []

        for method_name, detections in method_results.items():
            for detection in detections:
                detection["detection_method"] = method_name
                all_detections.append(detection)

        # Remove duplicates (tables detected by multiple methods)
        unique_detections = self.remove_duplicate_detections(all_detections)

        # Sort by page and confidence
        unique_detections.sort(key=lambda x: (x["page"], -x["confidence"]))

        return unique_detections

    def remove_duplicate_detections(self, detections: List[Dict]) -> List[Dict]:
        """Remove duplicate table detections based on overlap."""
        if not detections:
            return []

        unique = []

        for detection in detections:
            is_duplicate = False

            for existing in unique:
                if detection["page"] == existing["page"] and self.bboxes_overlap(
                    detection["bbox"], existing["bbox"]
                ):

                    # Keep the one with higher confidence
                    if detection["confidence"] > existing["confidence"]:
                        unique.remove(existing)
                        unique.append(detection)
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(detection)

        return unique

    def bboxes_overlap(
        self, bbox1: List[float], bbox2: List[float], threshold: float = 0.5
    ) -> bool:
        """Check if two bounding boxes overlap significantly."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate intersection
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap

        # Calculate areas
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        # Calculate overlap ratio
        union = area1 + area2 - intersection
        overlap_ratio = intersection / union if union > 0 else 0

        return overlap_ratio > threshold

    def extract_tables_with_camelot(self, table_detections: List[Dict]) -> List[Dict]:
        """Extract actual table data using Camelot on detected areas."""
        if not CAMELOT_AVAILABLE:
            return []

        extracted_tables = []

        print(
            f"\n🐪 Extracting {len(table_detections)} detected tables with Camelot..."
        )

        for i, detection in enumerate(table_detections):
            try:
                page = detection["page"]
                bbox = detection["bbox"]

                # Convert bbox to Camelot format (string)
                table_area = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

                # Extract table using Camelot
                tables = camelot.read_pdf(
                    self.pdf_path,
                    pages=str(page),
                    flavor="stream",  # Use stream for better detection
                    table_areas=[table_area],
                )

                if tables and len(tables) > 0:
                    table = tables[0]  # Take first table in area

                    extracted_tables.append(
                        {
                            "detection_index": i,
                            "page": page,
                            "bbox": bbox,
                            "detection_method": detection["detection_method"],
                            "detection_confidence": detection["confidence"],
                            "extraction_accuracy": table.accuracy,
                            "shape": table.df.shape,
                            "table_data": table.df.to_dict(),
                            "csv_data": table.df.to_csv(index=False),
                            "preview": (
                                table.df.head(3).to_dict() if not table.df.empty else {}
                            ),
                        }
                    )

                    print(
                        f"   ✅ Table {i+1}: {table.df.shape} (accuracy: {table.accuracy:.2f})"
                    )
                else:
                    print(f"   ❌ Table {i+1}: No data extracted")

            except Exception as e:
                print(f"   ❌ Table {i+1}: Extraction failed - {e}")

        return extracted_tables

    def generate_detection_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate summary of detection results."""
        summary = {
            "total_detections": len(results.get("combined_tables", [])),
            "pages_with_tables": len(
                set(t["page"] for t in results.get("combined_tables", []))
            ),
            "detection_methods_used": list(results.get("detection_methods", {}).keys()),
            "successful_extractions": 0,
            "avg_extraction_accuracy": 0,
            "method_performance": {},
        }

        # Analyze method performance
        for method, detections in results.get("detection_methods", {}).items():
            summary["method_performance"][method] = {
                "detections": len(detections),
                "avg_confidence": (
                    sum(d["confidence"] for d in detections) / len(detections)
                    if detections
                    else 0
                ),
            }

        # Analyze extractions if available
        if "extracted_tables" in results:
            extracted = results["extracted_tables"]
            summary["successful_extractions"] = len(extracted)

            if extracted:
                accuracies = [t["extraction_accuracy"] for t in extracted]
                summary["avg_extraction_accuracy"] = sum(accuracies) / len(accuracies)

        return summary

    def close(self):
        """Close the PDF document."""
        self.doc.close()


def main():
    """Test the custom table detector."""

    # Look for test PDFs
    pdf_files = list(Path(".").glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found to test!")
        print("Please add PDF files to the current directory")
        return

    # Test with first PDF
    test_pdf = pdf_files[0]
    print(f"🧪 Testing custom detection with: {test_pdf}")

    try:
        detector = CustomTableDetector(str(test_pdf))
        results = detector.detect_tables_all_methods()

        # Save results
        output_file = f"custom_table_detection_{Path(test_pdf).stem}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {output_file}")

        # Print summary
        summary = results["summary"]
        print(f"\n📊 DETECTION SUMMARY:")
        print(f"   Total detections: {summary['total_detections']}")
        print(f"   Pages with tables: {summary['pages_with_tables']}")
        print(f"   Successful extractions: {summary['successful_extractions']}")
        if summary["avg_extraction_accuracy"] > 0:
            print(f"   Average accuracy: {summary['avg_extraction_accuracy']:.2f}")

        print(f"\n🎯 METHOD PERFORMANCE:")
        for method, perf in summary["method_performance"].items():
            print(
                f"   {method}: {perf['detections']} detections (avg confidence: {perf['avg_confidence']:.2f})"
            )

        # Save extracted tables as CSV
        if "extracted_tables" in results and results["extracted_tables"]:
            tables_dir = Path(f"custom_extracted_tables_{Path(test_pdf).stem}")
            tables_dir.mkdir(exist_ok=True)

            for table in results["extracted_tables"]:
                csv_filename = (
                    f"table_page_{table['page']}_method_{table['detection_method']}.csv"
                )
                csv_path = tables_dir / csv_filename

                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write(table["csv_data"])

            print(f"\n📁 Extracted tables saved to: {tables_dir}")

        detector.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
