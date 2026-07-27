"""
fetch_chunk.py

EurLexChunker: chunks a EUR-Lex-style regulatory HTML document -- given
either a local HTML file or a URL -- into article/paragraph/table/footnote
units with hierarchy paths, validates coverage against the full document
body, and can save everything to a JSON file.

This module defines the class only -- see scripts/ for the per-document
runners (e.g. scripts/crr_chunk.py, scripts/gdpr_chunk.py).

Needs:  pip install requests beautifulsoup4
"""

import json
import re
from collections import Counter
from pathlib import Path

import requests
from bs4 import BeautifulSoup


class EurLexChunker:
    """
    Chunks a single EUR-Lex-style regulatory HTML document into hierarchical
    JSON chunks: recitals, articles/paragraphs (incl. quoted-article
    amendments and split-out definitions), data tables, annexes, and
    footnotes. Also validates the result's text coverage against the source
    document.

    `source` may be either a path to a local HTML file or an http(s) URL.
    When it's a URL, the document is downloaded once and cached at
    `cache_file` (reused on later runs instead of re-downloading); when it's
    a local file path, it's read directly.

    `levels`, `article_id_pattern` and `unit_word` let this be reused for
    non-EUR-Lex-drafted documents (e.g. an LLM-generated HTML rendering of a
    Basel/FATF/CSSF PDF) whose native structure doesn't match EU conventions
    -- extend `levels` with new type names for structural units EU documents
    don't have, relax `article_id_pattern` if the finest unit's label isn't
    purely numeric, and set `unit_word` so citations say what the source
    document actually calls that unit (e.g. "Principle" or "Rec.") instead of
    always claiming "Art." All three default to exactly today's CRR/GDPR
    behaviour, so existing callers are unaffected. The article-level id
    *prefix* itself ("art_") is always assumed literally -- it's an internal
    bookkeeping detail never shown to the user, unlike the label and the
    citation word, which are.

    `verbose` (default True) controls the chatty step-by-step printout
    (download/cache messages, structure discovery, per-step "found N X"
    counts, save confirmation). Regardless of `verbose`, the two signals that
    actually matter -- unmapped id prefixes to add to `levels`, and the final
    coverage percentage -- always print.
    """

    # id prefix -> level name (verify/extend using the discovery printout)
    DEFAULT_LEVELS = {
        "prt": "part", "tis": "title", "tit": "title",
        "cpt": "chapter", "sct": "section", "sbs": "sub-section", "anx": "annex",
    }

    # Regex matching article-level container ids, e.g. id="art_92" or
    # id="art_429a". Override only the label portion (after "art_") for
    # source documents whose finest unit isn't purely numeric.
    DEFAULT_ARTICLE_ID_PATTERN = r"^art_[0-9]+[a-z]*$"

    # ------------------------------------------------------------------
    # Second paragraph-numbering convention, seen on the "original Official
    # Journal" markup generation (as opposed to the "no-parag" span used by
    # consolidated/documentation-tool pages, e.g. CRR, CRD5, MAR, MIFID2_L2).
    # 16 of 21 Bucket A documents actually use this generation. Each
    # paragraph is its own <div id="002.001"> wrapping a
    # <p class="oj-normal">1.   text...</p> -- there's no span.no-parag
    # anywhere, so without this, _process_article() never finds a paragraph
    # boundary and the whole article collapses into one chunk (verified:
    # AIFMD/art_37 came out as one 28,228-char chunk). See
    # test_chunk_granularity.py, which guards against this specific failure
    # mode by comparing per-article paragraph counts in the HTML vs chunks.
    # ------------------------------------------------------------------
    OJ_PARA_DIV_ID_RE = re.compile(r"^\d+\.\d+[a-z]?$")
    OJ_PARA_LEADING_NUM_RE = re.compile(r"^(\d+[a-z]?)\.\s")

    # ------------------------------------------------------------------
    # definitions-list splitter regex.
    #
    # EU legislative drafting puts an entire "Definitions" article's terms in
    # ONE run-on paragraph: "...shall apply: (1) 'credit institution' means
    # ...; (2) 'investment firm' means ...; ...". There's no HTML marker
    # between term (1) and term (2) — the only signal is this repeating
    # inline pattern: a parenthesized number/label immediately followed by a
    # smart-quoted term. Sub-clause letters like (a), (b), (i) never precede
    # a quote character, so they don't false-match. Verified against CRR
    # Art. 4(1): 179 distinct definitions (gaps at repealed 4/12/79, plus
    # amendment-inserted labels like 14a/20a/52a-i), all bundled into one
    # ~9,100-word chunk without this split.
    # ------------------------------------------------------------------
    DEF_BOUNDARY = re.compile(r"[:;.]\s*(\(\d+[a-z]?\))\s*(?=[‘'])")

    def __init__(self, doc_id, source, cache_file=None, levels=None,
                 article_id_pattern=None, unit_word="Art.", request_timeout=120,
                 verbose=True):
        self.doc_id = doc_id
        self.source = source
        self.levels = levels or dict(self.DEFAULT_LEVELS)
        self.article_id_pattern = article_id_pattern or self.DEFAULT_ARTICLE_ID_PATTERN
        self._article_id_re = re.compile(self.article_id_pattern)
        self.unit_word = unit_word
        self.request_timeout = request_timeout
        self.verbose = verbose
        self.cache_file = Path(cache_file) if cache_file else Path("html_cache") / f"{doc_id.lower()}.html"

        self.soup = None
        self.chunks = []
        self.unknown_prefixes = set()

    @staticmethod
    def _is_url(source):
        return isinstance(source, str) and re.match(r"^https?://", source) is not None

    # ------------------------------------------------------------------
    # STEP 1: load — download once (reuses the local cache afterwards) if
    # `source` is a URL, otherwise read the local HTML file directly.
    # ------------------------------------------------------------------
    def _load_html(self):
        if self._is_url(self.source):
            if not self.cache_file.exists() or self.cache_file.stat().st_size == 0:
                if self.verbose:
                    print("downloading...")
                r = requests.get(self.source, headers={"User-Agent": "Mozilla/5.0"}, timeout=self.request_timeout)
                r.raise_for_status()
                if len(r.text) < 10_000:
                    raise RuntimeError(
                        f"downloaded content looks too small ({len(r.text)} chars) — "
                        "likely a blocked/empty response, not saving. First 500 chars:\n"
                        f"{r.text[:500]!r}"
                    )
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                self.cache_file.write_text(r.text, encoding="utf-8")
                if self.verbose:
                    print(f"saved {self.cache_file} ({len(r.text)/1e6:.1f} MB)")
            html_text = self.cache_file.read_text(encoding="utf-8")
        else:
            html_text = Path(self.source).read_text(encoding="utf-8")

        self.soup = BeautifulSoup(html_text, "html.parser")

    # ------------------------------------------------------------------
    # STEP 2: discovery — print every id family so you can check `levels`
    # ------------------------------------------------------------------
    def _print_discovery(self):
        print("\n--- structure found in document ---")
        counts = Counter()
        samples = {}
        for el in self.soup.find_all(id=True):
            prefix = re.split(r"[_.\d]", el["id"])[0]
            counts[prefix] += 1
            if prefix not in samples:
                samples[prefix] = (el["id"], el.get_text(" ", strip=True)[:60])
        for p, n in counts.most_common(15):
            print(f"  {p:8} x{n:5}  e.g. id={samples[p][0]!r}  text={samples[p][1]!r}")
        print("-----------------------------------\n")

    # ------------------------------------------------------------------
    # table helper — shared by the article loop and the annex loop.
    #
    # CRR uses <table> for two very different things (verified against the
    # real document, not assumed):
    #   - ~87% of tables have 1-2 rows: these are formula/variable-definition
    #     layouts ("h = the amount of..."), not real tabular data. Flattening
    #     them to plain text loses nothing, so they get folded into the
    #     surrounding paragraph exactly like before.
    #   - the rest have 3+ rows: real tabular data (risk-weight tables,
    #     correlation matrices, disclosure templates). These get their own
    #     chunk with rows/cells preserved, so "which number is in which
    #     column" survives — plus a flattened text version for full-text search.
    #
    # Row count alone isn't enough, though — heavily-amending directives
    # (verified against 5AMLD) lay out nested "(a)...(i)..." amendment
    # clauses using <table> purely for indentation, which also has >2 rows
    # but is really just ordinary paragraph text, not data. Real CRR data
    # tables have short, terse cell values (median 5 chars, max 151 across
    # 64 real tables checked); 5AMLD's amendment-layout tables have cells
    # up to 10,737 chars (a whole clause crammed into one "cell"). 400 is a
    # threshold comfortably above CRR's real max and comfortably below
    # 5AMLD's smallest oversized cell, so it separates the two cleanly
    # without needing column-count regularity (real CRR tables can
    # legitimately have irregular column counts too — merged headers,
    # staircase-shaped correlation matrices — so that signal isn't safe to
    # filter on).
    # ------------------------------------------------------------------
    MAX_DATA_CELL_CHARS = 400

    @staticmethod
    def _extract_table(table):
        # scoped to this table's own rows/cells only. find_all("tr")/("td")
        # are recursive by default, so a naive call also picks up rows and
        # cells belonging to any NESTED sub-table (amendment-clause tables
        # nest sub-tables 2-3 levels deep for indentation) -- counting the
        # same cell's text once per nesting level it's found at (verified
        # against 5AMLD: some clauses were tripled). A cell containing a
        # nested table still gets that nested table's text via its own
        # get_text() below, exactly once, so nothing is lost by excluding
        # the nested table's rows from this table's own row list.
        rows = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
        row_texts, structured_rows = [], []
        for tr in rows:
            cells = [c for c in tr.find_all(["td", "th"]) if c.find_parent("table") is table]
            cell_texts = [c.get_text(" ", strip=True) for c in cells]
            structured_rows.append(cell_texts)
            joined = " ".join(t for t in cell_texts if t)
            if joined:
                row_texts.append(joined)
        flattened = "; ".join(row_texts)
        max_cell_len = max((len(c) for row in structured_rows for c in row), default=0)
        is_data_table = len(rows) > 2 and max_cell_len <= EurLexChunker.MAX_DATA_CELL_CHARS
        return is_data_table, ({"rows": structured_rows} if is_data_table else None), flattened

    def _split_definitions(self, chunk):
        text = chunk.get("text", "")
        matches = list(self.DEF_BOUNDARY.finditer(text))
        if len(matches) < 5:
            return [chunk]

        starts = [m.start(1) for m in matches] + [len(text)]
        intro = text[:starts[0]].strip()

        result = []
        for i in range(len(starts) - 1):
            seg_text = text[starts[i]:starts[i + 1]].strip()
            m = re.match(r"\((\d+[a-z]?)\)\s*[‘']([^’']+)[’']", seg_text)
            label = m.group(1) if m else str(i + 1)
            term = m.group(2) if m else None
            if i == 0 and intro:
                seg_text = f"{intro} {seg_text}"
            result.append({
                "chunk_id": f'{chunk["chunk_id"]}/def_{label}',
                "doc_id": chunk["doc_id"],
                "path": chunk["path"] + [{"type": "definition", "label": label, "heading": term}],
                "citation": f'{chunk["citation"]}, point ({label})',
                "text": seg_text,
            })
        return result

    # ------------------------------------------------------------------
    # STEP 3A: Preamble/Recitals Chunking
    #
    # Two real conventions seen for numbered recitals: most documents put
    # the number and text in the same <p> ("(1) The protection of..."),
    # but heavily-amending directives (verified against 5AMLD) lay recitals
    # out as a two-column table instead — one <td> holds just the bare
    # "(1)", a sibling <td> in the same <tr> holds the actual text. Without
    # handling the second convention, the bare-number <p> still "matches"
    # the numbering pattern (with an empty remainder) and silently produces
    # a recital chunk containing only "(1)" — wrong, not just incomplete.
    # ------------------------------------------------------------------
    def _chunk_recitals(self):
        first_art = self.soup.find("div", id=self._article_id_re)
        if not first_art:
            return

        pre_art_p = []
        curr = first_art.find_previous("p")
        while curr:
            pre_art_p.append(curr)
            curr = curr.find_previous("p")
        pre_art_p.reverse()

        recital_count = 0
        for p in pre_art_p:
            p_text = p.get_text(" ", strip=True)
            # Match standard EU recital numbering like (1), (2), (12) etc.
            match = re.match(r"^[(（](\d+)[)）]\s*(.+)", p_text)
            if match:
                num = match.group(1)
                recital_text = p_text
            else:
                bare = re.match(r"^[(（](\d+)[)）]$", p_text)
                if not bare:
                    continue
                td = p.find_parent("td")
                sib_td = td.find_next_sibling("td") if td else None
                sib_text = sib_td.get_text(" ", strip=True) if sib_td else ""
                if not sib_text:
                    continue
                num = bare.group(1)
                recital_text = f"({num}) {sib_text}"

            self.chunks.append({
                "chunk_id": f"{self.doc_id}/recital_{num}",
                "doc_id": self.doc_id,
                "path": [
                    {"type": "preamble", "label": "Preamble", "heading": "Recitals"},
                    {"type": "recital", "label": num, "heading": None}
                ],
                "citation": f"{self.doc_id} Recital {num}",
                "text": recital_text,
            })
            recital_count += 1
        if recital_count > 0 and self.verbose:
            print(f"found and chunked {recital_count} recitals")

    # ------------------------------------------------------------------
    # STEP 3A-2: Citations Chunking
    #
    # Distinct from recitals: the "Having regard to the Treaty...", "Having
    # regard to the proposal from the Commission...", "Acting in accordance
    # with the ordinary legislative procedure..." legal-basis/procedural
    # references that precede the numbered recitals. Verified against AMLR:
    # each one is its own <div id="cit_N">, a real structural container our
    # code never looked for at all -- not a recital-numbering variant, a
    # wholly separate category that was silently dropped entirely.
    # ------------------------------------------------------------------
    def _chunk_citations(self):
        citation_divs = self.soup.find_all("div", id=re.compile(r"^cit_\d+$"))
        count = 0
        for div in citation_divs:
            text = self._norm(div.get_text(" ", strip=True))
            if not text:
                continue
            num = div["id"].split("_", 1)[1]
            self.chunks.append({
                "chunk_id": f"{self.doc_id}/citation_{num}",
                "doc_id": self.doc_id,
                "path": [
                    {"type": "preamble", "label": "Preamble", "heading": "Citations"},
                    {"type": "citation", "label": num, "heading": None}
                ],
                "citation": f"{self.doc_id} Citation {num}",
                "text": text,
            })
            count += 1
        if count > 0 and self.verbose:
            print(f"found and chunked {count} citations")

    # ------------------------------------------------------------------
    # STEP 3B: Article Chunking (incl. tables found inside articles)
    # ------------------------------------------------------------------
    def _chunk_articles(self):
        articles = self.soup.find_all("div", id=self._article_id_re)
        if self.verbose:
            print(f"found {len(articles)} articles")
        for art in articles:
            self._process_article(art)

    def _process_article(self, art):
        art_label = art["id"].split("_", 1)[1]           # "92", "429a", ...

        # build the path by climbing parent divs
        path = []
        for parent in reversed(art.find_parents("div", id=True)):
            last = parent["id"].split(".")[-1]          # e.g., "sct_1"
            prefix, _, label = last.partition("_")      # "sct", "1"
            if prefix in self.levels:
                # Get the heading for this structural division
                heading = None
                # Find the header element in direct children first
                for p in parent.find_all("p", recursive=False):
                    if p.get("class") and any(cls in p["class"] for cls in ["title-division-2", "title-annex-2"]):
                        heading = p.get_text(" ", strip=True)
                        break
                if not heading:
                    for p in parent.find_all("p", recursive=False):
                        if p.get("class") and any(cls in p["class"] for cls in ["title-division-1", "title-annex-1"]):
                            heading = p.get_text(" ", strip=True)
                            break
                if not heading:
                    # Fallback to recursive search
                    heading_el = (
                        parent.find("p", class_="title-division-2")
                        or parent.find("p", class_="title-annex-2")
                        or parent.find("p", class_="title-division-1")
                        or parent.find("p", class_="title-annex-1")
                    )
                    heading = heading_el.get_text(" ", strip=True) if heading_el else None

                path.append({
                    "type": self.levels[prefix],
                    "label": label,
                    "heading": heading,
                })
            elif prefix not in ("art",):
                self.unknown_prefixes.add(prefix)

        # article title and subtitle, e.g. "Scope" or "Capital instruments..."
        # oj-sti-art/oj-ti-art are the original-OJ-generation equivalents of
        # stitle-article-norm/title-article-norm (see OJ_PARA_DIV_ID_RE above).
        title_el = art.find("p", class_=re.compile("stitle-article-norm|stitle-article-quoted|oj-sti-art"))
        if not title_el:
            title_el = art.find("p", class_=re.compile("title-article-norm|title-article-quoted|oj-ti-art"))
        art_heading = title_el.get_text(" ", strip=True) if title_el else None

        path.append({"type": "article", "label": art_label, "heading": art_heading})

        # Group direct children of the article div to ensure we capture all text, including unnumbered paragraphs and list items
        children = [c for c in art.find_all(recursive=False) if c.name]

        current_chunk = None
        accumulated_texts = []
        table_counter = [0]  # list so render_child (closure) can mutate it

        def emit_current_chunk():
            if current_chunk:
                current_chunk["text"] = " ".join(accumulated_texts)
                self.chunks.extend(self._split_definitions(current_chunk))

        def render_child(child, owner_chunk):
            """Return child's text, extracting any table found *anywhere inside
            it* (not just direct children — tables are often wrapped 2-3 divs
            deep, e.g. div.norm > div.inline-element > div.centered > table,
            verified against art_121) into its own chunk, replaced by a short
            marker. Formula-layout tables (<=2 rows) are left in place and
            flatten into the surrounding text exactly like before.

            Also strips modref (amendment/correction marker, e.g. '▼M17') paragraphs
            found anywhere inside — not just direct children. 490 of 1546 modrefs in
            CRR are nested inside a paragraph wrapper (e.g. div.norm.inline-element),
            not a direct child of the article, so the top-level skip check in the
            main loop below never sees them; verified against the real document."""
            if child.name == "table":
                is_data_table, structured, flattened = self._extract_table(child)
                if not is_data_table:
                    return flattened
                table_counter[0] += 1
                n = table_counter[0]
                self.chunks.append({
                    "chunk_id": f'{owner_chunk["chunk_id"]}/table_{n}',
                    "doc_id": self.doc_id,
                    "path": owner_chunk["path"] + [{"type": "table", "label": str(n), "heading": None}],
                    "citation": f'{owner_chunk["citation"]} (Table {n})',
                    "table": structured,
                    "text": flattened,
                })
                return f"[see Table {n}]"

            nested_tables = child.find_all("table")
            nested_modrefs = child.find_all(class_="modref")
            if not nested_tables and not nested_modrefs:
                return child.get_text(" ", strip=True)

            child_copy = BeautifulSoup(str(child), "html.parser")
            for m in child_copy.find_all(class_="modref"):
                m.decompose()
            for orig_t, copy_t in zip(nested_tables, child_copy.find_all("table")):
                is_data_table, structured, flattened = self._extract_table(orig_t)
                if is_data_table:
                    table_counter[0] += 1
                    n = table_counter[0]
                    self.chunks.append({
                        "chunk_id": f'{owner_chunk["chunk_id"]}/table_{n}',
                        "doc_id": self.doc_id,
                        "path": owner_chunk["path"] + [{"type": "table", "label": str(n), "heading": None}],
                        "citation": f'{owner_chunk["citation"]} (Table {n})',
                        "table": structured,
                        "text": flattened,
                    })
                    copy_t.replace_with(f" [see Table {n}] ")
                # else: leave copy_t as-is, it flattens into the surrounding text below
            return child_copy.get_text(" ", strip=True)

        for child in children:
            # Skip title elements and modification references (modref) to keep text content clean
            if child.name == "p" and any(cls in child.get("class", []) for cls in ["title-article-norm", "stitle-article-norm", "title-article-quoted", "stitle-article-quoted", "oj-ti-art", "oj-sti-art", "modref"]):
                continue
            if child.name == "div" and any(cls in child.get("class", []) for cls in ["eli-title", "modref"]):
                continue

            # Check if this child starts a new paragraph. Two conventions:
            # (a) span.no-parag inside the child (consolidated pages), or
            # (b) the child itself is an oj-normal paragraph wrapper div
            #     (original OJ pages) -- see OJ_PARA_DIV_ID_RE above.
            span = child.find("span", class_="no-parag")
            oj_num_match = None
            if not span and child.name == "div" and self.OJ_PARA_DIV_ID_RE.match(child.get("id", "")):
                oj_p = child.find("p", class_="oj-normal", recursive=False)
                if oj_p:
                    oj_num_match = self.OJ_PARA_LEADING_NUM_RE.match(oj_p.get_text(" ", strip=True))

            if span or oj_num_match:
                # Emit the previous chunk first
                emit_current_chunk()
                accumulated_texts = []

                # Start a new paragraph chunk
                num = span.get_text(" ", strip=True).rstrip(".") if span else oj_num_match.group(1)

                # Check if this paragraph is inside a quoted article
                marker = span if span else oj_p
                prev_titles = marker.find_all_previous("p", class_=re.compile("^(title-article-norm|title-article-quoted)$"), limit=1)
                is_quoted = False
                quoted_label = None
                quoted_heading = None
                if prev_titles:
                    prev_title = prev_titles[0]
                    if "title-article-quoted" in prev_title.get("class", []):
                        is_quoted = True
                        quoted_label = prev_title.get_text(" ", strip=True).lstrip("‘'\"").split()[-1].rstrip("’'\"")

                        sub_el = prev_title.find_next("p", class_="stitle-article-quoted")
                        if sub_el:
                            next_title = prev_title.find_next("p", class_=re.compile("^(title-article-norm|title-article-quoted)$"))
                            is_between = False
                            for sib in prev_title.find_all_next():
                                if sib == sub_el:
                                    break
                                if sib == next_title:
                                    is_between = True
                                    break
                            if not is_between:
                                quoted_heading = sub_el.get_text(" ", strip=True).strip("‘'\"’")

                if is_quoted:
                    chunk_id = f"{self.doc_id}/art_{art_label}_art_{quoted_label}/para_{num}"
                    citation = f"{self.doc_id} {self.unit_word} {art_label} ({self.unit_word} {quoted_label}({num}))"
                    para_path = path + [
                        {"type": "article", "label": quoted_label, "heading": quoted_heading},
                        {"type": "paragraph", "label": num, "heading": None}
                    ]
                else:
                    chunk_id = f"{self.doc_id}/art_{art_label}/para_{num}"
                    citation = f"{self.doc_id} {self.unit_word} {art_label}({num})"
                    para_path = path + [{"type": "paragraph", "label": num, "heading": None}]

                current_chunk = {
                    "chunk_id": chunk_id,
                    "doc_id": self.doc_id,
                    "path": para_path,
                    "citation": citation,
                }
                # Append the text of the child (pulls out any nested data-table into its own chunk)
                child_text = render_child(child, current_chunk)
                if child_text:
                    accumulated_texts.append(child_text)
            else:
                # If no paragraph has started yet, start an "intro" chunk (or single-paragraph chunk)
                if current_chunk is None:
                    current_chunk = {
                        "chunk_id": f"{self.doc_id}/art_{art_label}",
                        "doc_id": self.doc_id,
                        "path": path,
                        "citation": f"{self.doc_id} {self.unit_word} {art_label}",
                    }
                child_text = render_child(child, current_chunk)
                if child_text:
                    accumulated_texts.append(child_text)

        # Emit the last chunk
        emit_current_chunk()

    # ------------------------------------------------------------------
    # STEP 3C: Annexes Chunking (incl. tables found inside annexes)
    #
    # AMLR's annex ids have a non-breaking space between the prefix and the
    # label ("anx_\xa0I", not "anx_I" -- verified against the real
    # document, invisible in a browser) which silently failed the old
    # strict id=[A-Za-z0-9]+ regex, dropping every annex. \s* tolerates
    # that (Python's \s matches \xa0); .strip() on the label removes it
    # from the label/citation text so it doesn't leak into "Annex \xa0I".
    # ------------------------------------------------------------------
    def _chunk_annexes(self):
        annexes = self.soup.find_all("div", id=re.compile(r"^(anx|annex)_\s*[A-Za-z0-9]+$"))
        if not annexes:
            return
        if self.verbose:
            print(f"found {len(annexes)} annexes")
        for anx in annexes:
            anx_id = anx["id"]
            anx_label = anx_id.split("_", 1)[1].strip()  # e.g., "I", "II"

            # Find Annex Heading
            heading_el = (
                anx.find("p", class_="title-annex-2")
                or anx.find("p", class_="title-annex-2", recursive=True)
            )
            if not heading_el:
                title1_el = anx.find("p", class_="title-annex-1")
                if title1_el:
                    heading_el = title1_el.find_next("p")

            anx_heading = heading_el.get_text(" ", strip=True) if heading_el else None

            # Copy and clean the annex HTML tree using BS4 deepcopy to avoid mutating original DOM
            anx_clean = BeautifulSoup(str(anx), "html.parser").find("div")
            for t in anx_clean.find_all("p", class_=re.compile("title-annex")):
                t.decompose()

            anx_path = [{"type": "annex", "label": anx_label, "heading": anx_heading}]

            # Pull out real data tables as their own chunks first, replacing each
            # with a short marker so the surrounding narrative text stays coherent.
            # Formula-layout tables (<=2 rows) are left in place and flatten
            # naturally into the narrative text below, same as before.
            anx_table_counter = 0
            for t in anx_clean.find_all("table"):
                is_data_table, structured, flattened = self._extract_table(t)
                if is_data_table:
                    anx_table_counter += 1
                    self.chunks.append({
                        "chunk_id": f"{self.doc_id}/anx_{anx_label}/table_{anx_table_counter}",
                        "doc_id": self.doc_id,
                        "path": anx_path + [{"type": "table", "label": str(anx_table_counter), "heading": None}],
                        "citation": f"{self.doc_id} Annex {anx_label} (Table {anx_table_counter})",
                        "table": structured,
                        "text": flattened,
                    })
                    t.replace_with(f" [see Table {anx_table_counter}] ")

            anx_text = anx_clean.get_text(" ", strip=True)
            if not anx_text:
                continue

            if len(anx_text) > 10000:
                elements = anx_clean.find_all("p")
                curr_chunk_text = []
                curr_len = 0
                part_num = 1

                for el in elements:
                    el_text = el.get_text(" ", strip=True)
                    if not el_text:
                        continue

                    if curr_len + len(el_text) > 4000 and curr_chunk_text:
                        self.chunks.append({
                            "chunk_id": f"{self.doc_id}/anx_{anx_label}/part_{part_num}",
                            "doc_id": self.doc_id,
                            "path": anx_path + [{"type": "annex_part", "label": str(part_num), "heading": None}],
                            "citation": f"{self.doc_id} Annex {anx_label} (Part {part_num})",
                            "text": " ".join(curr_chunk_text),
                        })
                        curr_chunk_text = []
                        curr_len = 0
                        part_num += 1

                    curr_chunk_text.append(el_text)
                    curr_len += len(el_text)

                if curr_chunk_text:
                    self.chunks.append({
                        "chunk_id": f"{self.doc_id}/anx_{anx_label}/part_{part_num}",
                        "doc_id": self.doc_id,
                        "path": anx_path + [{"type": "annex_part", "label": str(part_num), "heading": None}],
                        "citation": f"{self.doc_id} Annex {anx_label} (Part {part_num})",
                        "text": " ".join(curr_chunk_text),
                    })
            else:
                self.chunks.append({
                    "chunk_id": f"{self.doc_id}/anx_{anx_label}",
                    "doc_id": self.doc_id,
                    "path": anx_path,
                    "citation": f"{self.doc_id} Annex {anx_label}",
                    "text": anx_text,
                })

    # ------------------------------------------------------------------
    # STEP 3D: Footnote Chunking
    #
    # Verified against real documents: footnote <p> elements live at the
    # very top of <body>, outside every article/annex div — the article/
    # annex loops above never see them, so without this step they are
    # silently dropped (confirmed: 44/44 in CRR). Two real class-name
    # conventions exist (checked all 22 downloaded documents, not assumed):
    # older documents (CRR, GDPR, CRD5, MAR, ...) use class="footnote";
    # 13 of the 22 -- SFDR, DORA, AMLR, MiFID II, EMIR, etc. -- use
    # class="oj-note" instead, and were silently losing 100% of their
    # footnotes before this handled both.
    #
    # Both conventions share the same underlying structure, just different
    # naming, so one implementation covers both instead of special-casing
    # each: the footnote's first <a> carries its own id, a nested <span>
    # holds the visible number, and the <a>'s href points directly at the
    # citing location elsewhere in the document (old convention:
    # id="E0001" / href="#src.E0001"; new convention: id="ntr1-...-E0001"
    # / href="#ntc1-...-E0001" — different strings, same shape). Following
    # the href directly, rather than hardcoding either naming scheme,
    # works for both without needing to know which one a given document uses.
    # ------------------------------------------------------------------
    def _chunk_footnotes(self):
        footnote_ps = self.soup.find_all("p", class_=("footnote", "oj-note"))
        footnote_count = 0
        for fp in footnote_ps:
            anchor = fp.find("a", id=True)
            if not anchor or not anchor.get("href", "").startswith("#"):
                continue
            num_el = anchor.find("span")
            if num_el and num_el.get_text(strip=True):
                num = num_el.get_text(strip=True)
            else:
                num = re.sub(r"\D", "", anchor["id"]).lstrip("0") or "0"

            # only the self-referencing number anchor gets removed entirely
            # -- some conventions (SFDR's oj-note) also wrap the actual
            # citation text in a real hyperlink (e.g. a link to the Official
            # Journal page), which must stay as text, just without the link
            # wrapper, or the footnote's whole content is wiped out
            # (verified against SFDR: was reduced to a lone ".").
            fp_copy = BeautifulSoup(str(fp), "html.parser")
            first_a = fp_copy.find("a")
            if first_a:
                first_a.decompose()
            for a in fp_copy.find_all("a"):
                a.unwrap()
            text = fp_copy.get_text(" ", strip=True).lstrip("()（） ").strip()
            if not text:
                continue

            citing_id = anchor["href"].lstrip("#")
            src = self.soup.find(id=citing_id)
            citing_path = [{"type": "preamble", "label": "Preamble", "heading": "Citations"}]
            citing_citation = f"{self.doc_id} Preamble"
            if src:
                art_anc = src.find_parent("div", id=self._article_id_re)
                if art_anc:
                    anc_label = art_anc["id"].split("_", 1)[1]
                    citing_citation = f"{self.doc_id} {self.unit_word} {anc_label}"
                    citing_path = [{"type": "article", "label": anc_label, "heading": None}]

            self.chunks.append({
                "chunk_id": f"{self.doc_id}/footnote_{num}",
                "doc_id": self.doc_id,
                "path": citing_path + [{"type": "footnote", "label": num, "heading": None}],
                "citation": f"{self.doc_id} Footnote {num} (cited in {citing_citation})",
                "text": text,
            })
            footnote_count += 1
        if footnote_count and self.verbose:
            print(f"found and chunked {footnote_count} footnotes")

    # ------------------------------------------------------------------
    # STEP 4: validation — how much of the FULL document text did we capture?
    #
    # Checked against soup.body (everything on the page), not just the
    # enc_1 container, since footnotes live outside enc_1 entirely — checking
    # only enc_1 would silently hide the fact that they used to be dropped.
    # ------------------------------------------------------------------
    @staticmethod
    def _norm(s):
        return " ".join(s.split())

    def _validate(self):
        body = self.soup.body or self.soup
        original_len = len(self._norm(body.get_text(" ", strip=True)))
        captured_len = sum(len(self._norm(c["text"])) for c in self.chunks)
        coverage = captured_len / original_len
        print(f"coverage: {coverage:.1%} of full document body text captured "
              f"({captured_len:,} / {original_len:,} chars)")
        if self.verbose:
            print("(nav/TOC/language-switcher boilerplate is not chunked, so <100% is expected)")

        # sanity checks
        empty = [c["chunk_id"] for c in self.chunks if len(c["text"]) < 20]
        if empty:
            print(f"WARNING: {len(empty)} suspiciously short chunks, e.g. {empty[:3]}")

        dupes = len(self.chunks) - len({c["chunk_id"] for c in self.chunks})
        if dupes:
            print(f"WARNING: {dupes} duplicate chunk_ids")

    # ------------------------------------------------------------------
    # STEP 5: save
    # ------------------------------------------------------------------
    def _save(self, out_file):
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.chunks, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.verbose:
            print(f"\nsaved {len(self.chunks)} chunks -> {out_path}")
            print("sample chunk:")
            print(json.dumps(self.chunks[min(500, len(self.chunks) - 1)], indent=2)[:800])

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, out_file=None, validate=True):
        """
        Runs the full pipeline: load the document (from `source`, a local
        HTML file or a URL) -> discovery printout (if verbose) -> chunk
        recitals, articles, annexes and footnotes -> optional coverage
        validation -> optional save to `out_file`. Returns the list of
        chunks. Regardless of `verbose`, always prints the unmapped-prefix
        warning (if any) and the coverage line from `_validate`.
        """
        self._load_html()

        if self.verbose:
            self._print_discovery()

        self.chunks = []
        self.unknown_prefixes = set()

        self._chunk_citations()
        self._chunk_recitals()
        self._chunk_articles()
        self._chunk_annexes()
        self._chunk_footnotes()

        if self.unknown_prefixes:
            print(f"WARNING: unmapped id prefixes seen inside articles: {self.unknown_prefixes}")
            print("         add them to `levels` if structural (rerun with verbose=True to see "
                  "the discovery printout).\n")

        if self.verbose:
            print(f"created {len(self.chunks)} chunks")

        if validate:
            self._validate()

        if out_file:
            self._save(out_file)

        return self.chunks
