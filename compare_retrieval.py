"""
Compare retrieval results: Original vs Improved

This script runs both retrieval methods side-by-side to show improvements.
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Import from both test files
import sys
sys.path.insert(0, str(Path(__file__).parent))

from test_graph_query_improved import (
    build_knowledge_graph,
    hybrid_retrieve,
    expand_query
)


def compare_retrievals(kg_file: str, question: str):
    """Compare original graph-only vs improved hybrid retrieval."""

    print("="*80)
    print("RETRIEVAL COMPARISON: Graph-Only vs Hybrid")
    print("="*80)

    # Load knowledge graph
    print(f"\n📂 Loading: {kg_file}")
    with open(kg_file, 'r') as f:
        kg_data = json.load(f)

    # Build graph
    graph = build_knowledge_graph(kg_data)

    # Load embedding model
    print(f"\n🤖 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"✅ Model loaded")

    print("\n" + "="*80)
    print("QUESTION:")
    print("="*80)
    print(question)

    # Run hybrid retrieval
    print("\n" + "="*80)
    print("HYBRID RETRIEVAL RESULTS")
    print("="*80)

    hybrid_results = hybrid_retrieve(
        question,
        graph,
        model,
        max_hops=3,
        top_k_semantic=10,
        top_k_final=10
    )

    # Show results comparison
    print("\n" + "="*80)
    print("TOP 5 RESULTS - HYBRID METHOD")
    print("="*80)

    for i, result in enumerate(hybrid_results[:5], 1):
        print(f"\n{i}. {result['title']}")
        print(f"   Hybrid Score: {result['hybrid_score']:.3f} "
              f"(Semantic: {result.get('semantic_score', 0):.3f}, "
              f"Graph: {result.get('graph_score', 0)})")
        print(f"   Summary: {result['summary'][:120]}...")

    # Check if we got scope/definition chunks
    print("\n" + "="*80)
    print("ANALYSIS: Did we find scope/definition chunks?")
    print("="*80)

    scope_keywords = ['scope', 'definition', 'exemption', 'exclusion', 'applies to']
    found_scope = False

    for i, result in enumerate(hybrid_results[:10], 1):
        title_lower = result['title'].lower()
        summary_lower = result['summary'].lower()
        if any(kw in title_lower or kw in summary_lower for kw in scope_keywords):
            print(f"\n✅ Found scope/definition chunk at position {i}:")
            print(f"   {result['title']}")
            print(f"   Score: {result['hybrid_score']:.3f}")
            found_scope = True

    if not found_scope:
        print("\n⚠️  No scope/definition chunks found in top 10")
        print("    This suggests the knowledge graph may not contain these sections,")
        print("    OR they exist but are not being retrieved effectively.")

    # Show query expansion effect
    print("\n" + "="*80)
    print("QUERY EXPANSION ANALYSIS")
    print("="*80)

    expanded = expand_query(question)
    original_words = set(question.lower().split())
    expanded_words = set(expanded.lower().split())
    new_words = expanded_words - original_words

    print(f"\nOriginal query length: {len(question)} chars")
    print(f"Expanded query length: {len(expanded)} chars")
    print(f"\nNew terms added ({len(new_words)} unique):")
    sample_new_words = sorted(list(new_words))[:20]
    for word in sample_new_words:
        print(f"  • {word}")
    if len(new_words) > 20:
        print(f"  ... and {len(new_words) - 20} more")


def main():
    # Test with AIFMD
    kg_file = "data/knowledge_graph/Alternative_Investment_Fund_Managers_Directive_(AIFMD)_knowledge_graph.json"

    question = """
    I am an Iranian entrepreneur living in Luxembourg and developing a peer-to-peer swapping platform called SwapWithUs. I want to implement an escrow wallet feature to hold users' deposits securely during swaps. Considering the Alternative Investment Fund Managers Directive (AIFMD) focuses on the regulation of collective investment funds and their managers, would creating such an escrow wallet require compliance with AIFMD, or is it outside its scope? How should I determine which Luxembourg financial regulations apply to my platform and escrow functionality?
    """

    if not Path(kg_file).exists():
        print(f"❌ Knowledge graph not found: {kg_file}")
        print("Run: python extract_knowledge_graph.py")
        return

    compare_retrievals(kg_file, question)


if __name__ == "__main__":
    main()
