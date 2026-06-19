"""
Extract Knowledge Graph from Full Document

Processes document in batches (10 pages at a time) and extracts:
- Entities and relationships
- Summaries at multiple levels
- Builds complete knowledge graph

Saves incrementally to avoid data loss.
"""

import json
from pathlib import Path
from src.knowledge_graph.llm_extractor import LLMExtractor
from datetime import datetime


def read_pdf_in_batches(pdf_path: str, pages_per_batch: int = 10, max_batches: int = None):
    """
    Read PDF in batches.

    Args:
        pdf_path: Path to PDF
        pages_per_batch: Number of pages per batch

    Yields:
        (batch_num, batch_text, start_page, end_page)
    """
    try:
        import fitz
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        batch_num = 0
        current_page = 0

        while current_page < total_pages and (batch_num <= max_batches if max_batches is not None else True):
            # Get pages for this batch
            end_page = min(current_page + pages_per_batch, total_pages)

            text_parts = []
            for page_num in range(current_page, end_page):
                text_parts.append(doc[page_num].get_text())

            batch_text = " ".join(text_parts)

            yield (batch_num, batch_text, current_page + 1, end_page)

            batch_num += 1
            current_page = end_page

        doc.close()

    except Exception as e:
        print(f"Error reading PDF: {e}")
        raise


def save_batch_result(batch_result: dict, output_dir: Path):
    """
    Save a single batch result to file.

    Args:
        batch_result: Extraction result for one batch
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_file = output_dir / f"batch_{batch_result['batch_num']:03d}.json"

    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump(batch_result, f, indent=2, ensure_ascii=False)

    print(f"   ✅ Saved batch {batch_result['batch_num']} to {batch_file.name}")


def merge_all_batches(output_dir: Path, document_name: str):
    """
    Merge all batch results into a single knowledge graph.

    Args:
        output_dir: Directory containing batch files
        document_name: Name of the document

    Returns:
        Merged knowledge graph data
    """
    batch_files = sorted(output_dir.glob("batch_*.json"))

    if not batch_files:
        print("❌ No batch files found to merge!")
        return None

    print(f"\n📦 Merging {len(batch_files)} batches...")

    all_chunks = []
    all_entities = {}  # entity_id -> entity data (deduplicate)
    all_relationships = []

    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)

        # Add chunks
        all_chunks.extend(batch_data['chunks'])

        # Merge entities (deduplicate by ID)
        for entity in batch_data.get('all_entities', []):
            entity_id = entity['id']
            if entity_id not in all_entities:
                all_entities[entity_id] = entity
            # If entity exists, we could merge descriptions here if needed

        # Add relationships
        all_relationships.extend(batch_data.get('all_relationships', []))

    # Create merged result
    merged = {
        'document': document_name,
        'total_batches': len(batch_files),
        'total_chunks': len(all_chunks),
        'total_entities': len(all_entities),
        'total_relationships': len(all_relationships),
        'chunks': all_chunks,
        'entities': list(all_entities.values()),
        'relationships': all_relationships,
        'extraction_date': datetime.now().isoformat()
    }

    # Save merged result
    merged_file = output_dir.parent / f"{document_name}_knowledge_graph.json"
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"✅ Merged result saved to: {merged_file}")

    return merged


def main():
    print("="*80)
    print("EXTRACT KNOWLEDGE GRAPH FROM DOCUMENT")
    print("="*80)

    # Configuration
    pdf_file = "data/raw/regulatory_documents/eu/EBA_Guidelines_on_Limited_Network_Exclusion (Article 3(k) PSD2).pdf"
    document_name = Path(pdf_file).stem
    pages_per_batch = 10
    output_dir = Path(f"data/knowledge_graph/{document_name}_batches")

    print(f"\n📄 Document: {Path(pdf_file).name}")
    print(f"📊 Processing: {pages_per_batch} pages per batch")
    print(f"💾 Output: {output_dir}\n")

    # Initialize extractor
    extractor = LLMExtractor(model="gpt-4o-mini", use_azure=False)

    # Process document in batches
    print("="*80)
    print("PROCESSING BATCHES")
    print("="*80)

    batch_count = 0
    total_entities = 0
    total_relationships = 0

    for batch_num, batch_text, start_page, end_page in read_pdf_in_batches(pdf_file, pages_per_batch):

        print(f"\n{'─'*80}")
        print(f"📦 Batch {batch_num + 1}: Pages {start_page}-{end_page}")
        print(f"{'─'*80}")
        print(f"   Text length: {len(batch_text):,} characters")

        try:
            # Extract with LLM
            print(f"   🤖 Running LLM extraction...")
            result = extractor.extract(
                text=batch_text,
                document_name=f"{document_name} (Pages {start_page}-{end_page})",
                context="Regulatory/financial document"
            )

            # Collect ALL entities from this batch
            all_entities = list(result.global_entities)
            for chunk in result.chunks:
                all_entities.extend(chunk.entities)

            # Collect ALL relationships from this batch
            all_relationships = list(result.global_relationships)
            for chunk in result.chunks:
                all_relationships.extend(chunk.relationships)

            # Deduplicate entities by ID
            unique_entities = {}
            for entity in all_entities:
                if entity.id not in unique_entities:
                    unique_entities[entity.id] = entity

            print(f"   ✅ Extracted:")
            print(f"      Chunks: {len(result.chunks)}")
            print(f"      Unique Entities: {len(unique_entities)}")
            print(f"      Relationships: {len(all_relationships)}")

            # Prepare batch result
            batch_result = {
                'batch_num': batch_num,
                'pages': f"{start_page}-{end_page}",
                'document': document_name,
                'overall_summary': result.overall_summary,
                'chunks': [
                    {
                        'id': f"{document_name}_batch{batch_num}_{c.chunk_id}",
                        'batch': batch_num,
                        'title': c.title,
                        'summary': c.summary,
                        'key_points': c.key_points,
                        'content': c.content[:500],  # Save first 500 chars
                        'entities': [
                            {
                                'id': e.id,
                                'type': e.type,
                                'name': e.name,
                                'description': e.description,
                                'properties': e.properties
                            }
                            for e in c.entities
                        ],
                        'relationships': [
                            {
                                'source': r.source_entity_id,
                                'target': r.target_entity_id,
                                'type': r.relation_type,
                                'description': r.description,
                                'properties': r.properties
                            }
                            for r in c.relationships
                        ]
                    }
                    for c in result.chunks
                ],
                'all_entities': [
                    {
                        'id': e.id,
                        'type': e.type,
                        'name': e.name,
                        'description': e.description,
                        'properties': e.properties
                    }
                    for e in unique_entities.values()
                ],
                'all_relationships': [
                    {
                        'source': r.source_entity_id,
                        'target': r.target_entity_id,
                        'type': r.relation_type,
                        'description': r.description,
                        'properties': r.properties
                    }
                    for r in all_relationships
                ]
            }

            # Save batch immediately (don't lose data if it fails later)
            save_batch_result(batch_result, output_dir)

            batch_count += 1
            total_entities += len(unique_entities)
            total_relationships += len(all_relationships)

        except Exception as e:
            print(f"   ❌ Error processing batch {batch_num}: {e}")
            print(f"   Continuing with next batch...")
            continue

    # Merge all batches
    print("\n" + "="*80)
    print("MERGING BATCHES")
    print("="*80)

    merged = merge_all_batches(output_dir, document_name)

    if merged:
        print(f"\n" + "="*80)
        print("📊 FINAL STATISTICS")
        print("="*80)
        print(f"   Batches processed: {merged['total_batches']}")
        print(f"   Total chunks: {merged['total_chunks']}")
        print(f"   Total entities: {merged['total_entities']}")
        print(f"   Total relationships: {merged['total_relationships']}")
        print(f"\n✅ Knowledge graph extraction complete!")
        print("="*80)


if __name__ == "__main__":
    main()
