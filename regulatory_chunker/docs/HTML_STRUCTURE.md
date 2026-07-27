# EUR-Lex HTML Structure Reference

Everything below is verified directly against the real files (`crr.html`,
cross-checked against `gdpr.html` where noted), not assumed from
documentation. Where something is confirmed for CRR only, it's marked as
such — don't treat it as guaranteed for the other 21 documents in the
corpus without checking.

---

## 1. Document skeleton

```html
<html>
  <body>
    <div id="enc_1">          <!-- whole-document wrapper, not a real legal level -->
       ... recitals (no id scheme in CRR) ...
       <div id="prt_I"> ... all Parts/Titles/Chapters/.../Articles ... </div>
       <div id="anx_I"> ... Annex I ... </div>
    </div>
    <p class="footnote">...</p>   <!-- ALL footnotes, flat, OUTSIDE enc_1 -->
    <p class="footnote">...</p>
    ...
  </body>
</html>
```

Two things that trip you up if you don't know them going in:
- `enc_1` is not a legal structural unit — it's a bare container. The
  chunker deliberately does not map it to a `path` level (see §10).
- Footnotes are **not nested inside the article that cites them**. They sit
  as a flat list, disconnected from the rest of the document (see §7).

---

## 2. The ID scheme

### 2.1 Compound, dotted ids

Every structural div's `id` is built by **concatenating its own segment
onto all of its ancestors' ids**, joined with dots:

```html
<div id="prt_III">
  <div id="prt_III.tis_I">
    <div id="prt_III.tis_I.cpt_1">
      <div id="prt_III.tis_I.cpt_1.sct_1">
        <div id="art_92">...</div>
```

So the Section div's own id, `prt_III.tis_I.cpt_1.sct_1`, already encodes
Part III → Title I → Chapter 1 → Section 1. To get just *this* div's own
label, take the last dot-separated segment: `"sct_1"`.

Articles and annexes are the exception — their ids are **not** prefixed
with their ancestors' ids: `art_92`, `art_429a`, `anx_I`. Just the bare
prefix + label, regardless of how deep they're nested.

### 2.2 ID prefix meanings

| Prefix | Meaning | Example id | Compound? |
|---|---|---|---|
| `prt` | Part | `prt_III` | yes |
| `tis` | Title (division) | `prt_III.tis_I` | yes |
| `tit` | (also Title, seen as an anchor-only variant — see §5.2 trap) | — | yes |
| `cpt` | Chapter | `prt_III.tis_I.cpt_1` | yes |
| `sct` | Section | `...cpt_1.sct_1` | yes |
| `sbs` | Sub-section | `...sct_1.sbs_1` | yes |
| `anx` | Annex | `anx_I` | no |
| `art` | Article | `art_92`, `art_429a` | no |
| `enc` | Whole-document wrapper | `enc_1` | no — **not a legal level, deliberately unmapped** |

`prefix, _, label = last.partition("_")` is how the code splits e.g.
`"sct_1"` into `prefix="sct"`, `label="1"`.

**Trap**: `tis` (a real Title *division* wrapper) and `tit` (an article's
own heading-anchor id, e.g. `art_1.tit_1`) look similar but are unrelated —
`tit` here is not a Title-level structural div, don't confuse the two when
scanning id prefixes on a new document.

---

## 3. Structural hierarchy (Part / Title / Chapter / Section / Sub-section)

Every one of these levels — regardless of depth — is written with **at
most two heading lines**:

```html
<div id="prt_I">
   <p class="title-division-1">PART ONE</p>              <!-- bare label -->
   <p class="title-division-2">GENERAL PROVISIONS</p>     <!-- descriptive name -->
   <div id="prt_I.tis_I">
      <p class="title-division-1">TITLE I</p>
      <p class="title-division-2">SUBJECT MATTER, SCOPE AND DEFINITIONS</p>
      ...
```

Confirmed: **only `-1` and `-2` exist** — no `-3`/`-4`, in CRR or GDPR. This
isn't a coincidence of these two documents; it's because the class number
means "which of this division's (at most two) heading lines," not "which
depth level of the document." Depth is entirely handled by the id prefix
(§2.2), reused independently at every level — a document with 6 levels of
nesting still only ever needs `-1`/`-2` at each of those 6 levels, not
`-6`.

- `-1` (bare label) always appears **before** `-2` (descriptive name) in
  document order. The chunker deliberately searches for `-2` across all of
  a div's direct children *before* even looking for `-1`, specifically so
  the more informative name wins — see §11 for why merging these into one
  search silently breaks that preference.
- Heading search is `recursive=False` (direct children only) first, with a
  full recursive search only as a last-resort fallback — otherwise a
  parent div with no heading of its own could wrongly inherit a *nested
  child's* heading instead.

---

## 4. Annexes

Same two-line convention, parallel class names:

```html
<div id="anx_I">
   <p class="title-annex-1">ANNEX I</p>
   <p class="title-annex-2">Own funds disclosure templates</p>
   ...
```

Annexes do **not** use the `span.no-parag` paragraph-numbering convention
that articles use (§5.3) — they're chunked differently: real data tables
get pulled out first, then the remaining narrative text is either kept
whole (if short) or split by raw character count (~4,000 chars/chunk, if
the annex is long) since there's no natural per-provision boundary marker
to split on.

---

## 5. Articles

### 5.1 Skeleton

```html
<div id="art_2" class="eli-subdivision">
   <p class="title-article-norm">Article 2</p>
   <div class="eli-title">
      <p class="stitle-article-norm">Supervisory powers</p>
   </div>
   <div class="norm">
      <span class="no-parag">1.</span>
      <div class="norm inline-element">For the purpose of ensuring compliance...</div>
   </div>
   <div class="norm">
      <span class="no-parag">2.</span>
      <div class="norm inline-element">For the purpose of ensuring compliance with the procedures...</div>
   </div>
   ...
```

### 5.2 Two different title classes — not a duplicate

| Class | Content | Example |
|---|---|---|
| `title-article-norm` | Bare label | `"Article 2"` |
| `stitle-article-norm` ("**s**ubtitle") | Actual descriptive heading | `"Supervisory powers"` |
| `title-article-quoted` / `stitle-article-quoted` | Same pair, but for an article being **inline-quoted verbatim inside a different article** (e.g. an amending regulation reproducing another article's full replacement text) | — |

The chunker prefers `stitle-*` (the real heading), falling back to
`title-*` only if no subtitle exists.

### 5.3 Numbered paragraphs — the only chunk-boundary signal

There is **no distinguishing class** between paragraph 1's wrapper div and
paragraph 2's — both are just `<div class="norm">`. The *only* marker that
a new numbered paragraph starts is a nested:

```html
<span class="no-parag">1.</span>
```

sitting as the first child inside the wrapper. This is why the chunker's
paragraph-splitting logic specifically searches *inside* each child for
this span, rather than checking the child's own class/name.

### 5.4 Quoted articles

Detected not by DOM nesting but by scanning backward from a `no-parag`
paragraph to the nearest preceding article-title line: if that title was
`title-article-quoted` rather than `title-article-norm`, the paragraph
belongs to a *different* article number being quoted inline, and gets
chunked as `art_{outer}_art_{quoted}/para_{n}` instead of colliding with
the real article of that number elsewhere in the document. Confirmed: no
`art_` div is ever physically nested inside another `art_` div in CRR —
this quoting is purely a text/heading-sequence pattern, not a DOM-nesting
one.

---

## 6. Tables

```html
<table>
   <tr><td>...</td><td>...</td></tr>
   ...
</table>
```

Two very different uses of `<table>`, distinguished purely by row count:

| Row count | What it actually is | Handling |
|---|---|---|
| ≤ 2 rows | Formula / variable-definition layout (`"h = the amount of..."`), not real tabular data | Flattened into the surrounding paragraph text, no separate chunk |
| > 2 rows | Real tabular data (risk-weight tables, disclosure templates, correlation matrices) | Extracted into its own `.../table_N` chunk with structured `rows`, plus a flattened text version |

Tables are frequently nested 2-3 divs deep inside an article's paragraph
(e.g. `div.norm > div.norm.inline-element > div.centered > table` inside
`art_121`) rather than being a direct child — any table-detection logic
must search recursively, not just check direct children.

---

## 7. Footnotes

```html
<!-- near the front of <body>, disconnected from any article -->
<p class="footnote">(<a href="#src.E0001" id="E0001"><span class="superscript">1</span></a>) Directive 2014/59/EU...</p>
```

```html
<!-- inline, wherever the footnote is actually cited, deep inside some article -->
...set out in Directive 2014/59/EU (<a href="#E0001" id="src.E0001"><span class="superscript">1</span></a>) and in this Regulation.
```

Two mirror-image anchors joined only by a shared numeric suffix:
- `id="E0001"` — the footnote's own identity, on the footnote paragraph.
- `id="src.E0001"` — the citation point, inline in whatever article cited
  it.

To recover *which article* a footnote belongs to, you have to find the
`src.E0001` anchor and climb its ancestors to the nearest `art_` div — the
footnote text itself lives nowhere near that article in the DOM.

---

## 8. Amendment / correction markers (`modref`)

```html
<p class="modref">▼M17</p>
```

Provenance annotations (`▼M17` = amended by the 17th amending act, `▼C2` =
corrected by the 2nd corrigendum) — not legal text, always excluded from
chunk text. **Trap**: ~1/3 of these in CRR are nested inside a wrapper div
(e.g. `div.norm.inline-element`) rather than being a direct child of the
article, so a skip-check that only examines direct children will miss
them — must search recursively, exactly like table detection.

---

## 9. Definitions lists — an inline convention with no HTML marker at all

A "Definitions" article (e.g. Article 4) writes dozens or hundreds of
terms as **one continuous run of prose**, with no per-term markup:

```html
<div class="norm">
   <span class="no-parag">1.</span>
   <div class="norm inline-element">
      For the purposes of this Regulation, the following definitions shall apply:
      (1) ‘credit institution’ means an undertaking...; (2) ‘investment firm’ means...; (3) ‘institution’ means...
   </div>
</div>
```

The only reliable signal that a new definition starts is **a parenthesized
number/label immediately followed by a smart-quoted term** —
`(N) '<term>' means`. Sub-clause letters like `(a)`, `(b)`, `(i)` inside a
definition's own conditions never appear immediately before a quote
character, so they don't false-match. Confirmed in CRR: this convention
appears in 9 articles (4, 5, 5a, 142, 192, 242, 272, 300, 411), and label
numbers have gaps (repealed provisions) and letter-suffixed insertions
(`14a`, `20a`, `52a`–`52i`, etc.) from a decade of amendments — don't
assume the highest label number equals the total count of definitions.

---

## 10. What's confirmed across documents vs. CRR-specific

| Element | Confirmed in | Not yet checked |
|---|---|---|
| `art_`/`anx_` id scheme, compound dotted ids | CRR, GDPR | other 15 Bucket A docs |
| `title-division-1`/`-2` only (no `-3`+) | CRR, GDPR | other 15 Bucket A docs |
| `title-annex-1`/`-2` | CRR | GDPR (none found — may have no annexes with headings, or differ) |
| Recitals with no native id scheme | CRR only | other docs may use a real `rct_` prefix instead of needing the regex-on-preceding-paragraphs hack |
| `modref` nesting trap | CRR | assume present elsewhere until checked |
| Definitions-list inline convention | CRR (9 articles) | very likely present in any doc with a "Definitions" article (GDPR Art. 4, MiFID2 Art. 4, etc.) — same EU legislative drafting convention, not CRR-specific |

Before running the chunker against a new Bucket A document, re-run the
Step 2 discovery printout (id-prefix counts) and spot-check for any of
these classes appearing with an unexpected suffix or in an unexpected
place — don't assume this reference transfers 1:1 without checking.
