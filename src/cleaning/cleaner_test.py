import re

class RegulationCleaner:
    def __init__(self, raw_text: str):
        self.raw_text = raw_text

    def clean(self) -> str:
        text = self.raw_text
        text = self._remove_table_of_contents(text)
        text = self._remove_dotted_lines(text)
        text = self._normalize_whitespace(text)
        return text

    def _remove_table_of_contents(self, full_doc_text):
        """
        Removes a Table of Contents with the specified pattern
        (e.g., "Part I. ... 8", "Chapter 1. ... 11") from the document text.
        """
        # 1. Define markers for the start and potential end of the TOC content
        toc_header_pattern = r"TABLE OF CONTENTS" # Case-sensitive based on your example

        # This regex captures lines that look like ToC entries:
        # Starts with (Part|Chapter|Sub-chapter|Section)
        # Followed by a number/roman numeral, then text
        # Followed by dots, then a page number at the end of the line
        toc_line_pattern = r"(Part|Chapter|Sub-chapter|Section|ANNEX|APPEBDIX)\s+[\dIVXLCDM]+\..+?\s*\.{3,}\s*\d+"

        # Markers for the start of the *actual* content (not just a ToC line)
        # These should NOT have the dots and page numbers at the end.
        content_start_markers = [
            r"Part\s+[IVXLCDM]+\.",    # e.g., "Part I. Definitions..."
            r"Chapter\s+\d+\.",      # e.g., "Chapter 1. Basic principles"
            r"Sub-chapter\s+\d+\.\d+\.", # e.g., "Sub-chapter 2.1. Initial authorisation" (if it appears on its own line)
            r"Section\s+\d+\.\d+\.\d+\." # e.g., "Section 4.1.1. Required number"
        ]

        # Find the main "TABLE OF CONTENTS" header
        toc_header_match = re.search(toc_header_pattern, full_doc_text)


        if not toc_header_match:
            print("Warning: 'TABLE OF CONTENTS' header not found. Returning original text.")
            return full_doc_text

        # Start search for content after the TOC header
        search_start_index = toc_header_match.end()

        # Find the last line that matches a ToC entry pattern after the header
        last_toc_line_end_index = -1
        # Iterate through matches starting from the TOC header
        for match in re.finditer(toc_line_pattern, full_doc_text[search_start_index:]):
            last_toc_line_end_index = search_start_index + match.end()

        if last_toc_line_end_index == -1:
            print("Warning: No TOC-like lines found after 'TABLE OF CONTENTS' header. Returning original text.")
            return full_doc_text

        # Now, find the first *actual* content marker after the last ToC line
        # This is the most critical part for accurate trimming
        actual_content_start_index = -1
        for marker_pattern in content_start_markers:
            # Search specifically for the full string pattern, not just the start of a line
            # And ensure it's after the last detected ToC line
            match = re.search(marker_pattern, full_doc_text[last_toc_line_end_index:], re.IGNORECASE)
            if match:
                # Found a content marker. This is the beginning of the actual content.
                actual_content_start_index = last_toc_line_end_index + match.start()
                # If multiple markers match, take the earliest one.
                break # Exit loop once the first content start is found

        if actual_content_start_index != -1:
            print(f"Removed TOC from index {toc_header_match.start()} to {actual_content_start_index}")
            return full_doc_text[actual_content_start_index:]
        else:
            print("Warning: Could not reliably identify the start of actual content after the TOC. Attempting to trim after the last TOC line.")
            # Fallback: if no clear content marker, just trim after the last identified TOC line.
            # This might leave some blank lines or transitional text if not perfect.
            return full_doc_text[last_toc_line_end_index:]
    

    def _remove_dotted_lines(self, text: str) -> str:
        # Remove lines that are just dots, dashes, or mixed
        return re.sub(r"^[\s.\-•_]{5,}$", "", text, flags=re.MULTILINE)

    def _normalize_whitespace(self, text: str) -> str:
        # Normalize spaces and newlines
        text = re.sub(r"\n{2,}", "\n\n", text)  # Limit consecutive newlines
        text = re.sub(r"[ \t]+", " ", text)     # Normalize spaces
        return text.strip()
