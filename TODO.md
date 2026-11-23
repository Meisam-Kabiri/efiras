# TODO: fix the 3 PDF documents (BASEL3, FATF2012, CSSF_18_698)

They are almost missing from the index (9, 1 and 5 chunks), so questions about them get wrong answers.

- [ ] Fix `regulatory_chunker/tree_chunker.py`: add `prt`, `tit`, `sbs` to `levels` (and allow a `levels` argument).
- [ ] Build `html_cache/basel3.html` and `html_cache/fatf2012.html` from `regulatory_chunker/events_cache/` (free).
- [ ] Create the CSSF events + `html_cache/cssf_18_698.html` from `data/raw/regulatory_documents/lu/Lux_cssf18_698eng.pdf` (Claude API, paid).
- [ ] Chunk the 3 HTML files into `chunks_output/`.
- [ ] Re-embed and rebuild `data/regulatory_indexes/`, then push to deploy.
