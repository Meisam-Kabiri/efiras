# TODO

## Cross-reference limitations (core/rag/references.py)

- [ ] **Only one hop.** If a retrieved article refers to Article 429, Article 429 is added, but the
      articles that Article 429 itself refers to (e.g. Article 430) are not.
- [ ] **Basel III paragraph references are not followed.** The Basel III text refers to its own
      paragraph numbers ("paragraph 129", "paragraphs 154 to 164"). Only "Article N" references and
      FATF "Recommendation N" references are followed today.
- [ ] **References to other laws are skipped.** "Article 4(1) of Directive 2013/36/EU" is ignored,
      because there is no mapping from the act's name/number to a document in the index
      (e.g. Directive 2013/36/EU -> CRD5, Regulation (EU) No 575/2013 -> CRR).
- [ ] **Referenced articles are picked by position, not relevance.** Up to 6 articles are added, taken
      from the top-ranked chunks first. When a chunk refers to many articles, some added ones may be
      off-topic; the answer prompt tells the model to ignore them, but they still use prompt tokens.
- [ ] **Long referenced articles are cut.** Each referenced article is capped at about 1,000 tokens
      (~4,000 characters); later paragraphs are omitted.
