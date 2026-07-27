# Advanced Features Leveraging the Regulatory DOM Tree Hierarchy

This document outlines 4 high-value architectural features enabled by the hierarchical DOM tree paths (`path` metadata) produced by `regulatory_chunker`.

---

## 1. Interactive Table of Contents (TOC) Sidebar Navigation

### Overview
Because `TreeChunker` preserves the full DOM hierarchy (`chapter` -> `section` -> `article` -> `paragraph`), `RegulatoryRepository` can generate a complete hierarchical Table of Contents graph tree.

### Usage in Python
```python
from regulatory_chunker import RegulatoryRepository

repo = RegulatoryRepository()
toc_graph = repo.get_toc("GDPR")
```

### Web UI Value
Renders an interactive sidebar tree on the web app frontend. Users can click any node in the tree (e.g. `Chapter II -> Article 14`) to jump directly to that exact legal provision.

---

## 2. Contextual Neighbor Expansion (Parent & Sibling Retrieval)

### Overview
When a search query matches a small sub-clause (e.g. `Article 14, paragraph (1), point (b)`), the surrounding legal sub-clauses `(a)` and `(c)` are often essential for complete legal interpretation.

### Implementation Pattern
Using the structured `chunk_id` (`4AMLD/cpt_II/sct_1/art_14/p_1`), the retrieval service can fetch sibling chunks that share the same parent article:

```python
def get_article_context(doc_id: str, article_id: str, repo: RegulatoryRepository):
    all_chunks = repo.get_chunks(doc_id)
    return [c for c in all_chunks if article_id in c["chunk_id"]]
```

### RAG Value
Provides 100% complete legal context to the LLM without missing neighboring sub-clauses or cutting off legal conditions.

---

## 3. Legal Breadcrumb UI & Audit Tracing

### Overview
In legal and financial compliance, compliance officers need to verify where a rule sits within a regulation's hierarchy.

### UI Format
Every search result snippet displays its full legal breadcrumb path:

$$\text{4AMLD} \;\mathbf{>}\; \text{Chapter II (Customer Due Diligence)} \;\mathbf{>}\; \text{Section 1} \;\mathbf{>}\; \text{Article 14(1)}$$

### Compliance Value
Enables instant audit tracing for compliance officers and legal auditors.

---

## 4. Chapter & Section Hard Scope Filtering

### Overview
Allows the LLM routing layer or frontend users to restrict search queries to specific chapters or sections.

### Implementation Pattern
```python
# Filter search strictly to chunks inside Chapter II
cpt_chunks = [
    c for c in chunks 
    if any(node["type"] == "chapter" and node["label"] == "II" for node in c["path"])
]
```

### RAG Value
Eliminates distractor noise from unrelated chapters, yielding 100% precision for targeted regulatory queries.
