import re
from typing import List, Dict

class RegulationChunker:
    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.chunks = []

    def chunk(self) -> List[Dict]:
        # Combine all heading patterns
        heading_pattern = re.compile(
            r"^(Part\s+\w+.*?)$|^(Chapter\s+\d+.*?)$|^(Sub-chapter\s+\d+\.\d+.*?)$|^(Section\s+\d+\.\d+\.\d+.*?)$",
            re.MULTILINE
        )
        matches = list(heading_pattern.finditer(self.raw_text))

        for i, match in enumerate(matches):
            # Get the heading and its position
            heading = next(g for g in match.groups() if g is not None)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(self.raw_text)
            content = self.raw_text[start:end].strip()

            if content:  # Only store non-empty content
                self.chunks.append({
                    "heading": heading.strip(),
                    "content": content,
                    "hierarchy": self._parse_hierarchy(heading)
                })

        return self.chunks

    def _parse_hierarchy(self, heading: str) -> List[str]:
        """Turn heading into a hierarchy list like ['Part II', 'Chapter 4', ...]"""
        levels = []
        if heading.startswith("Part"):
            levels.append(heading)
        elif heading.startswith("Chapter"):
            levels.append(heading)
        elif heading.startswith("Sub-chapter"):
            levels.append(heading)
        elif heading.startswith("Section"):
            levels.append(heading)
        return levels

