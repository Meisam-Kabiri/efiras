#!/usr/bin/env python3
"""
Alternative: Use chunk-level pairing with better semantic similarity
"""

import json
import re
from typing import List, Dict
from pathlib import Path


class ChunkLevelMNRLBuilder:
    def __init__(self):
        # Key terms for similarity matching
        self.key_terms = {
            'risk_management': ['risk management', 'risk assessment', 'risk monitoring', 'risk control'],
            'compliance': ['compliance', 'regulatory requirement', 'obligation'],
            'governance': ['governance', 'board', 'director', 'management body'],
            'capital': ['capital', 'own funds', 'capital adequacy'],
            'delegation': ['delegation', 'delegate', 'third party'],
            'valuation': ['valuation', 'fair value', 'pricing'],
            'reporting': ['reporting', 'disclosure', 'notification'],
            'authorization': ['authorization', 'approval', 'permit']
        }
    
    def extract_key_concepts(self, text: str) -> set:
        """Extract key concepts from text"""
        text_lower = text.lower()
        concepts = set()
        
        for concept, terms in self.key_terms.items():
            if any(term in text_lower for term in terms):
                concepts.add(concept)
        
        return concepts
    
    def calculate_chunk_similarity(self, chunk1: Dict, chunk2: Dict) -> float:
        """Calculate semantic similarity between two chunks"""
        text1 = chunk1.get('text', '').lower()
        text2 = chunk2.get('text', '').lower()
        
        # Concept-based similarity
        concepts1 = self.extract_key_concepts(text1)
        concepts2 = self.extract_key_concepts(text2)
        
        if not concepts1 or not concepts2:
            concept_sim = 0
        else:
            concept_sim = len(concepts1 & concepts2) / len(concepts1 | concepts2)
        
        # Word-based similarity (excluding common regulatory terms)
        stopwords = {'the', 'of', 'and', 'to', 'in', 'a', 'an', 'ifm', 'cssf', 'law', 'regulation', 'article'}
        
        words1 = set(w for w in text1.split() if len(w) > 3 and w not in stopwords)
        words2 = set(w for w in text2.split() if len(w) > 3 and w not in stopwords)
        
        if not words1 or not words2:
            word_sim = 0
        else:
            word_sim = len(words1 & words2) / len(words1 | words2)
        
        # Combined similarity (weight concepts more heavily)
        return 0.7 * concept_sim + 0.3 * word_sim
    
    def group_chunks_by_topic(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """Group chunks by their primary topic"""
        groups = {}
        
        for chunk in chunks:
            concepts = self.extract_key_concepts(chunk.get('text', ''))
            
            if not concepts:
                primary_topic = 'general'
            else:
                # Use the first concept as primary topic
                primary_topic = list(concepts)[0]
            
            if primary_topic not in groups:
                groups[primary_topic] = []
            groups[primary_topic].append(chunk)
        
        return groups
    
    def create_high_quality_pairs(self, chunks: List[Dict], min_similarity: float = 0.3) -> List[Dict]:
        """Create high-quality pairs based on semantic similarity"""
        pairs = []
        used_indices = set()
        
        print(f"Finding high-quality pairs from {len(chunks)} chunks...")
        
        for i in range(len(chunks)):
            if i in used_indices:
                continue
            
            best_match = None
            best_similarity = min_similarity
            best_j = None
            
            for j in range(i + 1, len(chunks)):
                if j in used_indices:
                    continue
                
                similarity = self.calculate_chunk_similarity(chunks[i], chunks[j])
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = chunks[j]
                    best_j = j
            
            if best_match is not None:
                pairs.append({
                    'anchor': chunks[i]['text'][:500],
                    'positive': best_match['text'][:500],
                    'similarity': best_similarity,
                    'concepts_i': self.extract_key_concepts(chunks[i]['text']),
                    'concepts_j': self.extract_key_concepts(best_match['text'])
                })
                used_indices.add(i)
                used_indices.add(best_j)
        
        print(f"Created {len(pairs)} high-quality pairs")
        return pairs
    
    def generate_chunk_level_mnrl_dataset(self, chunks: List[Dict], batch_size: int = 8) -> List[List[Dict]]:
        """Generate MNRL dataset using chunk-level semantic pairing"""
        print(f"Generating chunk-level MNRL dataset...")
        
        # Filter valid chunks (skip very short ones and definition sections)
        valid_chunks = []
        for chunk in chunks:
            text = chunk.get('text', '')
            if len(text.strip()) < 100:
                continue
            # Skip definition sections
            if '"' in text and 'means' in text.lower():
                continue
            # Skip pure lists
            if len(re.findall(r'[a-z]\)\s+', text)) > 3:
                continue
            valid_chunks.append(chunk)
        
        print(f"Using {len(valid_chunks)} valid chunks (filtered from {len(chunks)})")
        
        # Create high-quality pairs
        pairs = self.create_high_quality_pairs(valid_chunks, min_similarity=0.35)
        
        # Create batches
        batches = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            if len(batch) == batch_size:
                # Convert to MNRL format
                mnrl_batch = []
                for pair in batch:
                    mnrl_batch.append({
                        'anchor': pair['anchor'],
                        'positive': pair['positive']
                    })
                batches.append(mnrl_batch)
        
        print(f"Created {len(batches)} complete batches")
        return batches, pairs  # Return pairs for validation
    
    def validate_pairs(self, pairs: List[Dict], num_samples: int = 10):
        """Validate the quality of pairs"""
        print(f"\n=== CHUNK PAIR QUALITY CHECK ===")
        
        similarities = [p['similarity'] for p in pairs]
        avg_sim = sum(similarities) / len(similarities) if similarities else 0
        
        print(f"Average similarity: {avg_sim:.3f}")
        print(f"Similarity range: {min(similarities):.3f} - {max(similarities):.3f}")
        
        for i, pair in enumerate(pairs[:num_samples]):
            print(f"\nPair {i+1} (Similarity: {pair['similarity']:.3f}):")
            print(f"  Shared concepts: {pair['concepts_i'] & pair['concepts_j']}")
            print(f"  Anchor: {pair['anchor'][:80]}...")
            print(f"  Positive: {pair['positive'][:80]}...")


def main():
    builder = ChunkLevelMNRLBuilder()
    
    # Load chunks
    file_path = "data_processed/Lux_cssf18_698eng_chunked_blocks.json"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Generate dataset
    batches, pairs = builder.generate_chunk_level_mnrl_dataset(chunks, batch_size=8)
    
    if batches:
        # Validate pairs
        builder.validate_pairs(pairs)
        
        # Save results
        output_file = f"training_data/mnrl_chunk_level_{Path(file_path).stem.replace('_chunked_blocks', '')}.json"
        Path(output_file).parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(batches, f, indent=2, ensure_ascii=False)
        
        total_pairs = len(batches) * 8
        print(f"\n✅ Saved {len(batches)} batches ({total_pairs} pairs) to {output_file}")
        print("🎯 Expected: Higher similarity scores and better training!")


if __name__ == "__main__":
    main()