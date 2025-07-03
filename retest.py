import re

def remove_toc_specific_pattern(full_doc_text):
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

# --- Example Usage ---
# Dummy text simulating your PDF content
pdf_text_example = """
Some initial document title or cover page text...
Some more preamble...

TABLE OF CONTENTS
Part I. Definitions and abbreviations .............................................................................................. 8
Part II. Conditions for obtaining and maintaining the authorisation of an authorised
investment fund manager (IFM) who engages solely in the activity of management of UCIs as
laid down in Article 101(2) of the 2010 Law and Article 5(2) of the 2013 Law ......................... 11
Chapter 1. Basic principles ............................................................................................................ 11
Chapter 2. Shareholding ............................................................................................................... 11
Sub-chapter 2.1. Initial authorisation ....................................................................................... 11
Sub-chapter 2.2. Changes in the shareholding ......................................................................... 13
Chapter 3. Own funds..................................................................................................................... 13
Sub-chapter 3.1. Required own funds ....................................................................................... 13
Sub-chapter 3.2. Eligible capital................................................................................................ 15
Sub-chapter 3.3. Use of own funds ............................................................................................ 15
Chapter 4. The bodies of the IFM ................................................................................................. 17
Sub-chapter 4.1. The members of the governing body or management body....................... 17
Section 4.1.1. Required number ........................................................................................................17
Section 4.1.2. Requirements regarding the skills, experience and good repute and the composition of
the management body/governing body .............................................................................................17
Section 4.1.3. Conditions for performing multiple mandates ...........................................................18
Section 4.1.4. Obligations regarding meetings and deliberations .....................................................19
Sub-chapter 4.2. Senior management ....................................................................................... 19
Section 4.2.1. Required number, presence in Luxembourg and contractual relationship with the IFM
ANNEXES .............................................................. 87
ANNEX 1: The risk management procedure of AIFs to be communicated to the CSSF ......... 87
.........................................................................................................................................................19
---PAGE_BREAK---
Part I. Definitions and abbreviations
For the purpose of this circular, the following definitions shall apply:
"2010 Law" means the Law of 17 December 2010 on undertakings for collective investment.
"2013 Law" means the Law of 12 July 2013 on alternative investment fund managers.
"CSSF" means the Commission de Surveillance du Secteur Financier.
"Depositary" means a depositary as defined in Article 3(1) of Directive 2011/61/EU.

Chapter 1. Basic principles
This chapter outlines the fundamental principles guiding the authorization and operation of IFMs...
"""

cleaned_text = remove_toc_specific_pattern(pdf_text_example)
print("--- Cleaned Text (Start) ---")
print(cleaned_text[:500]) # Print first 500 characters to verify
print("\n--- Cleaned Text (End) ---")