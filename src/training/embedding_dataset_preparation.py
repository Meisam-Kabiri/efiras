#!/usr/bin/env python3
"""
Embedding Dataset Preparation for EFIRAS
Automatically generate training pairs from regulatory documents
"""

import json
import random
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any
from itertools import combinations, cycle
from src.utils.text_utils import extract_header_identifier

class AdaptiveMNRLDatasetBuilder:
    def __init__(self):
        # No need for default paths - we'll take explicit file path
        
        # Financial terminology patterns for similarity detection
        self.financial_terms = {
            "capital": ["capital", "tier 1", "tier 2", "capital ratio", "capital adequacy", "minimum capital"],
            "liquidity": ["liquidity", "LCR", "liquid assets", "cash outflows", "HQLA"],
            "risk": ["risk", "credit risk", "market risk", "operational risk", "stress test"],
            "compliance": ["compliance", "regulatory", "mandatory", "required", "obligation"],
            "reporting": ["reporting", "disclosure", "publication", "submission", "filing"],
            "governance": ["governance", "management", "oversight", "control", "framework"]
        }
        
        # Regulation types for cross-regulation similarity
        self.regulation_types = {
            "basel": ["basel", "bcbs", "bank capital", "banking supervision"],
            "mifid": ["mifid", "investment services", "market conduct"],
            "cssf": ["cssf", "luxembourg", "circular", "regulation"],
            "emir": ["emir", "derivatives", "clearing", "margin"],
            "psd2": ["psd2", "payment services", "strong authentication"]
        }
        
        self.doc_type = None
    
    
                        
    def identify_regulation_type(self, text: str) -> str:
        """Identify which regulation family this text belongs to"""
        text_lower = text.lower()
        
        # Fixed: Check ALL regulation types and return the one with most matches
        best_match = "general"
        max_matches = 0
        
        for reg_type, keywords in self.regulation_types.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > max_matches:
                max_matches = matches
                best_match = reg_type
        
        return best_match
    
    def extract_financial_concepts(self, text: str) -> List[str]:
        """Extract financial concepts from text"""
        text_lower = text.lower()
        concepts = []
        
        for concept, keywords in self.financial_terms.items():
            if any(keyword in text_lower for keyword in keywords):
                concepts.append(concept)
        
        return concepts

    def extract_lowest_level_key(self, header: str) -> str:
        """Extract the lowest/deepest level from hierarchical header"""
        if not header:
            return "unknown"
        
        # Split by common separators (>, →, /)
        levels = re.split(r'\s*>\s*|\s*→\s*|\s*/\s*', header.strip())
        levels = [extract_header_identifier(lev) for lev in levels if extract_header_identifier(lev) is not None]
        if levels:
            return levels[-1].strip()  # Return the last/deepest level
        return header.strip()
    
    def group_by_lowest_level(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group chunks by their lowest hierarchy level ONLY (no overlap)
        Example: 'Part I > Chapter 2 > Section 3' → goes to 'Section 3' group only
        """
        level_groups = {}
        
        for chunk in chunks:
            header = chunk.get('enriched_headers', '')
            if not header:
                continue
                
            lowest_level = self.extract_lowest_level_key(header)
            
            if lowest_level not in level_groups:
                level_groups[lowest_level] = []
            level_groups[lowest_level].append(chunk)
        
        print(f"Grouped into {len(level_groups)} distinct hierarchy levels (no overlap)")
        for level, chunks in level_groups.items():
            print(f"  {level}: {len(chunks)} chunks")
        

        return level_groups
    
    
    def create_pairs_within_group(self, chunks: List[Dict], group_key: str) -> List[Dict]:
        """Create positive pairs within a homogeneous group"""
        pairs = []
        
        if len(chunks) < 2:
            return pairs
        
        # Determine pairing strategy based on group size
        if len(chunks) <= 5:
            # Small groups: pair everything with everything
            max_distance = len(chunks)
        else:
            # Large groups: limit to nearby chunks (more efficient)
            max_distance = 3
        
        pairs_created = 0
        max_pairs = len(chunks) * 2  # Reasonable limit
        
        for i in range(len(chunks) - 1):
            if pairs_created >= max_pairs:
                break
                
            for j in range(i + 1, min(i + max_distance, len(chunks))):
                chunk1, chunk2 = chunks[i], chunks[j]
                pairs.append({
                    'anchor': chunk1['text'][:500],
                    'positive': chunk2['text'][:500],
                    'group_key': group_key,
                    'hierarchy_level': self.determine_hierarchy_level(group_key)
                })
                pairs_created += 1
                
                if pairs_created >= max_pairs:
                    break
        
        return pairs
    
    def determine_hierarchy_level(self, group_key: str) -> int:
        """Determine hierarchy level from group key for correct sorting"""
        key_lower = group_key.lower()
        
        # Level 1: Part (e.g., "Part I", "Part II")
        if key_lower.startswith('part '):
            return 1
        
        # Level 2: Chapter (e.g., "Chapter 1", "Chapter 2") 
        elif key_lower.startswith('chapter ') and 'sub-chapter' not in key_lower:
            return 2
        
        # Level 3: Sub-chapter (e.g., "Sub-chapter 2.1", "Sub-chapter 3.1")
        elif 'sub-chapter' in key_lower:
            return 3
        
        # Level 4: Section (e.g., "Section 4.1.1", "Section 5.1.2")
        elif key_lower.startswith('section ') and 'sub-section' not in key_lower:
            return 4
        
        # Level 5: Sub-section (e.g., "Sub-section 5.3.1.1", "Sub-section 5.3.1.2")
        elif 'sub-section' in key_lower:
            return 5
        
        elif 'article' in key_lower:
            return 6
        
        # Level 6: Other/specific terms (e.g., "governing body", "CFT", etc.)
        else:
            return 7
    
    def create_fixed_size_batches(self, all_pairs: Dict[str, List[Dict]], batch_size: int = 8) -> List[List[Dict]]:
        """
        Create fixed-size batches maintaining hierarchy order
        Remove exhausted groups but preserve original order of remaining groups
        """
        # Sort groups by hierarchy level for consistent order
        sorted_groups = sorted(all_pairs.keys(), 
                              key=lambda x: (self.determine_hierarchy_level(x), x))
        
        # Create ordered dict to maintain hierarchy order
        grouped_chunks = {group: all_pairs[group].copy() for group in sorted_groups if all_pairs[group]}
        
        print(f"Creating fixed-size batches maintaining hierarchy order")
        print(f"Initial groups: {list(grouped_chunks.keys())}")
        
        batches = []
        
        # Continue until no more complete batches can be formed
        while True:
            # Get available groups (those with chunks remaining)
            available_groups = {k: v for k, v in grouped_chunks.items() if v}
            
            if len(available_groups) < 2:  # Need at least 2 groups for diversity
                break

            # Reshuffle the available group just for the same level
            group_keys = sorted(available_groups.keys(), 
                   key=lambda x: (self.determine_hierarchy_level(x), random.random()))

            if len(group_keys) >= batch_size:
                # Use only batch_size number of groups
                selected_keys = group_keys[:batch_size]
            else:
                # Cycle through available groups to fill batch
                key_cycle = cycle(group_keys)
                selected_keys = [next(key_cycle) for _ in range(batch_size)]
                
            
            # Create batch by taking one chunk from each selected group
            batch = []
            for key in selected_keys:
                if grouped_chunks[key]:
                    chunk = grouped_chunks[key].pop(0)
                    batch.append(chunk)
            
            # Only add batch if it's complete
            if len(batch) == batch_size:
                batches.append(batch)
                if len(batches) % 20 == 0:
                    remaining_groups = len([k for k, v in grouped_chunks.items() if v])
                    print(f"Created {len(batches)} batches, {remaining_groups} groups still active")
            else:
                break
        
        print(f"✅ Created {len(batches)} fixed-size batches maintaining hierarchy order")
        return batches
    
    def generate_fixed_size_mnrl_batches(self, chunks: List[Dict], batch_size: int = 8) -> List[List[Dict[str, str]]]:
        """
        Create MNRL dataset with fixed-size batches using advanced algorithm
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'enriched_headers' fields
            batch_size: Number of pairs per batch
        
        Algorithm:
        1. Group chunks by lowest hierarchy level only (no overlap)
        2. Create pairs within each homogeneous group  
        3. Use cyclic sampling to create diverse fixed-size batches
        4. Discard incomplete batches
        """
        print(f"Creating fixed-size MNRL dataset...")
        print(f"Target batch size: {batch_size}")
        
        # Process chunks
        blocks = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict) and len(chunk.get('text', '').strip()) >= 30:
                processed_chunk = chunk.copy()
                processed_chunk['chunk_index'] = i
                processed_chunk['financial_concepts'] = self.extract_financial_concepts(chunk.get('text', ''))
                processed_chunk['content_length'] = len(chunk.get('text', ''))
                processed_chunk['regulation_type'] = self.identify_regulation_type(chunk.get('text', ''))
                blocks.append(processed_chunk)
        
        if len(blocks) < 20:
            print("Warning: Very few blocks found. Need at least 20 blocks for meaningful pairs.")
            return []
        
        # Step 1: Group by lowest level only (eliminates overlap at source)
        level_groups = self.group_by_lowest_level(blocks)
        
        # Step 2: Create pairs within each homogeneous group
        all_pairs = {}
        total_pairs = 0
        
        for group_key, group_chunks in level_groups.items():
            pairs = self.create_pairs_within_group(group_chunks, group_key)
            if pairs:
                all_pairs[group_key] = pairs
                total_pairs += len(pairs)
                print(f"Group '{group_key}': {len(pairs)} pairs")
        
        print(f"Total pairs generated: {total_pairs}")
        
        # Step 3: Create fixed-size batches using cyclic sampling
        fixed_batches = self.create_fixed_size_batches(all_pairs, batch_size)
        
        # Convert to final MNRL format (remove metadata)
        mnrl_batches = []
        for batch in fixed_batches:
            mnrl_batch = []
            for pair in batch:
                mnrl_batch.append({
                    'anchor': pair['anchor'],
                    'positive': pair['positive']
                })
            mnrl_batches.append(mnrl_batch)
        
        final_pairs = sum(len(batch) for batch in mnrl_batches)
        print(f"✅ Fixed-size MNRL dataset: {len(mnrl_batches)} batches with {final_pairs} total pairs")
        print(f"Batch size consistency: All batches have exactly {batch_size} pairs")
        print("✅ Maximum diversity guaranteed within each batch")
        
        return mnrl_batches


if __name__ == "__main__":
    # Create dataset builder
    builder = AdaptiveMNRLDatasetBuilder()
    
    # Load chunks from file 
    file_path = "data_processed/Lux_cssf18_698eng_chunked_blocks.json"
    file_path ="data_processed/Basel_III_chunked_blocks.json"
    
    
    # Load chunks list
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks from {file_path}")
    
    # Create fixed-size MNRL dataset with correct hierarchy (RECOMMENDED)
    print("=== Creating Fixed-Size MNRL Dataset ===")
    mnrl_batches = builder.generate_fixed_size_mnrl_batches(chunks, batch_size=8)
    
    if mnrl_batches:
        # Save fixed-size MNRL dataset
        output_file = f"training_data/mnrl_fixed_batch_dataset_{Path(file_path).stem.replace('_chunked_blocks', '')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mnrl_batches, f, indent=2, ensure_ascii=False)
        print(f"Fixed-size MNRL dataset saved to {output_file}")
        
        total_pairs = sum(len(batch) for batch in mnrl_batches)
        print(f"Generated {len(mnrl_batches)} batches with {total_pairs} total training pairs")
        print("✅ Dataset optimized for fixed-size batch MNRL training with correct hierarchy")
    else:
        print("No dataset created - check your input chunks!")