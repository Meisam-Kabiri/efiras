"""
tree_to_markdown.py

Renders a GraphStructureBuilder structure (or a whole corpus of them) as an
indented outline, so it can be sanity-checked by eye -- same purpose as
chunks_to_markdown.py, but for the collapsed structure shape instead of the
flat chunk list.
"""

from pathlib import Path


def _render_children(children, depth, lines):
    for node in children:
        label = f"{node['type']} {node['label']}"
        if node["heading"]:
            label += f" — {node['heading']}"
        n = len(node["chunks"])
        if n:
            label += f"  [{n} own chunk{'s' if n != 1 else ''}]"
        lines.append("  " * depth + f"- {label}")
        _render_children(node["children"], depth + 1, lines)


def tree_to_markdown(tree, out_file):
    lines = []
    _render_children(tree["children"], 0, lines)
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def corpus_to_markdown(corpus, out_file):
    lines = []
    for doc_id, tree in corpus.items():
        lines.append(f"# {doc_id}\n")
        _render_children(tree["children"], 0, lines)
        lines.append("")
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
