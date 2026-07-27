"""
tree_chunker.py
DOM Tree Chunker for EUR-Lex HTML & LLM-generated HTML.
Traverses structural <div> hierarchies (chapters, sections, articles, paragraphs)
and outputs clean, citation-aware JSON chunks.
"""

from typing import List, Dict, Any, Tuple, Optional
from bs4 import BeautifulSoup
import re


class TreeChunker:
    """
    Parses structured EUR-Lex HTML (or PDF-to-HTML conversion output)
    by walking the DOM tree recursively and preserving legal hierarchy paths.
    """

    def __init__(self, doc_id: str):
        self.doc_id = doc_id

    levels = {
        "cpt": "chapter",
        "sct": "section",
        "art": "article",
        "prg": "paragraph",
        "sub": "subparagraph",
        "def": "definition",
        "anx": "annex",
        "app": "appendix",
    }

    DEF_BOUNDARY = re.compile(
        r"(?<=\.\s)"
        r"(?="
        r"(?:\(\d+[a-z]?\)|\([a-z]\))\s+"
        r"(?:['\u2018\u2019\u201c\u201d\"][^'\u2018\u2019\u201c\u201d\"]+['\u2018\u2019\u201c\u201d\"]\s+(?:means|shall|is|are|refers)|"
        r"(?:[A-Z][a-zA-Z0-9\s,\-\(\)]+?)\s+(?:means|shall|is|are|refers))"
        r")"
    )

    def _parse_id(self, div_id: str) -> Tuple[Optional[str], str]:
        if not div_id:
            return None, ""
        last = div_id.split(".")[-1]
        prefix, _, label = last.partition("_")
        return self.levels.get(prefix), label

    HEADING_CLASSES = {
        "heading",
        "title-division-1",
        "title-division-2",
        "title-annex-1",
        "title-annex-2",
        "oj-ti-section-1",
        "oj-ti-section-2",
        "oj-ti-art",
    }

    HEADING_PREFERENCE = [
        "heading",
        "title-division-2",
        "title-annex-2",
        "title-division-1",
        "title-annex-1",
        "oj-ti-section-2",
        "oj-ti-section-1",
    ]

    NON_CONTENT_CLASSES = HEADING_CLASSES | {
        "title-article-norm",
        "stitle-article-norm",
        "title-article-quoted",
        "stitle-article-quoted",
        "eli-title",
        "modref",
    }

    def _child_text(self, child, strip_marker=False):
        copy = BeautifulSoup(str(child), "html.parser")
        for m in copy.find_all(class_="modref"):
            m.decompose()
        if strip_marker:
            for s in copy.find_all("span", class_="no-parag"):
                s.decompose()
        return self._clean_text(copy.get_text(" "))

    def _heading(self, div):
        candidates = []
        div_id = div.get("id", "")
        for p in div.find_all("p"):
            parent_div = p.find_parent("div")
            if parent_div == div or (parent_div and parent_div.get("id", "").startswith(div_id + ".")):
                classes = set(p.get("class") or [])
                matched = classes & self.HEADING_CLASSES
                if matched:
                    text = self._clean_text(p.get_text(" "))
                    if text and text not in candidates:
                        candidates.append(text)

        if candidates:
            return " - ".join(candidates)
        return None

    def _direct_text(self, div):
        parts = []
        for child in div.children:
            if getattr(child, "name", None) in ("p", "div"):
                child_classes = set(child.get("class") or [])
                if child.name == "p" and (child_classes & self.NON_CONTENT_CLASSES):
                    continue
                if child.name == "div" and (child_classes & {"eli-title", "table-wrapper"}):
                    continue
                text = self._child_text(child)
                if text:
                    parts.append(text)
        return " ".join(parts)

    def _split_paragraphs(self, div) -> List[Tuple[Optional[str], str]]:
        paras = []
        current_label = None
        current_parts = []

        for child in div.children:
            if getattr(child, "name", None) in ("p", "div"):
                child_classes = set(child.get("class") or [])
                if child.name == "p" and (child_classes & self.NON_CONTENT_CLASSES):
                    continue
                if child.name == "div" and (child_classes & {"eli-title", "table-wrapper"}):
                    continue

                marker = child.find("span", class_="no-parag") if hasattr(child, "find") else None
                if marker:
                    marker_text = self._clean_text(marker.get_text())
                    clean_marker = marker_text.rstrip(".").strip("()")
                    if current_parts or current_label is not None:
                        text = " ".join(current_parts).strip()
                        if text:
                            paras.append((current_label, text))
                    current_label = clean_marker
                    current_parts = [self._child_text(child, strip_marker=True)]
                else:
                    text = self._child_text(child)
                    if text:
                        current_parts.append(text)

        if current_parts or current_label is not None:
            text = " ".join(current_parts).strip()
            if text:
                paras.append((current_label, text))

        return paras

    def _structural_children(self, div) -> List[Any]:
        results = []
        for child in div.find_all("div", recursive=False):
            child_id = child.get("id") or ""
            if "." in child_id or child_id.startswith(("cpt_", "sct_", "art_", "prg_", "sub_", "def_", "anx_", "app_")):
                results.append(child)
        return results

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text or "")
        return text.strip()

    def _build_citation(self, path: List[Dict[str, Any]]) -> str:
        art_node = next((node for node in path if node["type"] == "article"), None)
        prg_node = next((node for node in path if node["type"] == "paragraph"), None)
        sub_node = next((node for node in path if node["type"] == "subparagraph"), None)

        if not art_node:
            return self.doc_id

        citation = f"{self.doc_id} Art. {art_node['label']}"
        if prg_node and prg_node["label"]:
            citation += f"({prg_node['label']})"
        if sub_node and sub_node["label"]:
            citation += f"({sub_node['label']})"
        return citation

    def chunk(self, html_content: str) -> List[Dict[str, Any]]:
        soup = BeautifulSoup(html_content, "html.parser")
        body = soup.find("body") or soup
        chunks = []

        root_divs = self._structural_children(body)
        if not root_divs:
            root_divs = body.find_all("div", recursive=True)

        for root in root_divs:
            self._walk(root, [], chunks)

        return chunks

    def _walk(self, div, current_path: List[Dict[str, Any]], chunks: List[Dict[str, Any]]):
        div_id = div.get("id", "")
        level_type, label = self._parse_id(div_id)

        if not level_type:
            for child in self._structural_children(div):
                self._walk(child, current_path, chunks)
            return

        heading = self._heading(div)
        node = {
            "type": level_type,
            "label": label,
            "heading": heading,
        }
        new_path = current_path + [node]

        children = self._structural_children(div)

        if level_type == "article" and not children:
            paras = self._split_paragraphs(div)
            if len(paras) > 1:
                for p_label, p_text in paras:
                    p_node = {"type": "paragraph", "label": p_label or "", "heading": None}
                    p_path = new_path + [p_node]
                    citation = self._build_citation(p_path)
                    chunks.append(
                        {
                            "chunk_id": f"{div_id}/p_{p_label}" if p_label else div_id,
                            "doc_id": self.doc_id,
                            "path": p_path,
                            "citation": citation,
                            "text": p_text,
                        }
                    )
                return
            elif len(paras) == 1 and self.DEF_BOUNDARY.search(paras[0][1]):
                sub_defs = self.DEF_BOUNDARY.split(paras[0][1])
                for idx, sub_txt in enumerate(sub_defs, start=1):
                    sub_txt = sub_txt.strip()
                    if not sub_txt:
                        continue
                    m = re.match(r"^\((\d+[a-z]?|[a-z])\)", sub_txt)
                    def_label = m.group(1) if m else str(idx)
                    d_node = {"type": "definition", "label": def_label, "heading": None}
                    d_path = new_path + [d_node]
                    citation = self._build_citation(d_path)
                    chunks.append(
                        {
                            "chunk_id": f"{div_id}/def_{def_label}",
                            "doc_id": self.doc_id,
                            "path": d_path,
                            "citation": citation,
                            "text": sub_txt,
                        }
                    )
                return

        if not children:
            direct_text = self._direct_text(div)
            if direct_text:
                citation = self._build_citation(new_path)
                chunks.append(
                    {
                        "chunk_id": div_id,
                        "doc_id": self.doc_id,
                        "path": new_path,
                        "citation": citation,
                        "text": direct_text,
                    }
                )
        else:
            intro_text = self._direct_text(div)
            if intro_text:
                citation = self._build_citation(new_path)
                chunks.append(
                    {
                        "chunk_id": f"{div_id}/intro",
                        "doc_id": self.doc_id,
                        "path": new_path,
                        "citation": citation,
                        "text": intro_text,
                    }
                )
            for child in children:
                self._walk(child, new_path, chunks)
