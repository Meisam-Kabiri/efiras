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
from src.utils.text_utils import extract_header_identifier, extract_sentences

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import re

def extract_sentences_for_positive_pairs(chunk_text):
   # Step 1: Split by periods, but be careful with abbreviations
   sentences = re.split(r'\.(?=\s+[A-Z])', chunk_text)
   
   # Step 2: Handle special cases
   sentences = clean_regulatory_sentences(sentences)
   
   # Step 3: Filter quality sentences
   quality_sentences = []
   for sent in sentences:
       if is_valid_sentence(sent):
           quality_sentences.append(sent.strip())
   
   return quality_sentences

def clean_regulatory_sentences(sentences):
   cleaned = []
   for sent in sentences:
       # Remove bullet point artifacts
       sent = re.sub(r'^[•\-\*]\s*', '', sent)
       
       # Handle line breaks in middle of sentences
       sent = re.sub(r'\n+', ' ', sent)
       
       # Clean up spacing
       sent = re.sub(r'\s+', ' ', sent).strip()
       
       # Skip pure enumeration items
       if not re.match(r'^[a-z]\)\s*$', sent.strip()) and sent.strip():
           cleaned.append(sent)
   
   return cleaned

def is_valid_sentence(sentence):
   # Filter criteria
   word_count = len(sentence.split())
   return (
       word_count >= 10 and  # Skip very short sentences
       word_count <= 100 and  # Skip overly long sentences
       not sentence.strip().endswith(':') and  # Skip header-like sentences
       not re.match(r'^\d+\.\s*$', sentence.strip())  # Skip numbered items only
   )

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
    
    def create_sentence_level_pairs(self, chunks: List[Dict], group_key: str = '') -> List[Dict]:
        all_pairs = {}
        for chunk in chunks:
            content = chunk['text']
            sentence_list = extract_sentences_for_positive_pairs(content)
            sentences = clean_regulatory_sentences(sentence_list)
            sentences = [sent for sent in sentences if is_valid_sentence(sent)]

            n = len(sentences) 
            if n < 2:
                continue
            
            header = self.extract_lowest_level_key(chunk['enriched_headers'])
            # Debug unknown headers
            if header == 'unknown':
                print(f"UNKNOWN HEADER FOUND:")
                print(f"Raw header: '{chunk['enriched_headers']}'")
                print(f"Chunk ID: {chunk.get('chunk_id', 'N/A')}")
                print(f"Text preview: {chunk['text'][:100]}...")
                print("-" * 50)
            if header not in all_pairs.keys():
                all_pairs[header] = []
            # Adjacent pairs (always)

            for i in range(n - 1):
                all_pairs[header].append((sentences[i], sentences[i + 1]))
            # First-last pair (if 3+ sentences)  
            if n >= 3:
                all_pairs[header].append((sentences[0], sentences[-1]))
            
            # Skip-one pairs (if 4+ sentences)
            if n >= 4:
                for i in range(n - 2):
                    all_pairs[header].append((sentences[i], sentences[i + 2]))

                # if len(sentences)<2:
                    #     continue
                    # else:
                    #     # pair = (sentences[0], sentences[1])
                    #     model_transfomer =  SentenceTransformer('all-mpnet-base-v2', cache_folder='./models/')  # or your preferred model
                    #     # embd1 = model_transfomer.encode(pair[0])
                    #     # embd2 = model_transfomer.encode(pair[1])
                    #     # similarity = cosine_similarity([embd1], [embd2])[0][0]
                    #     # if similarity < 1:
                    #     #     print(f"1):{pair[0]}\n ----------------------\n 2:{pair[1]}")
                    #     #     print(f"cosine simalrty score is: {similarity} \n %%%%%%%%%%%%%%%%%%%%%%%\n")

        print(all_pairs.keys())
        return all_pairs

    def create_contrastive_positive_negative_pairs(self, chunks):
        contrastive_positive_pairs = []
        flattened_pairs = []
        all_pairs = self.create_sentence_level_pairs(chunks)
        for k, val in all_pairs.items():
            for v in val:
                contrastive_positive_pairs.append({
                    'anchor': v[0],
                    'positive': v[1],
                    'label': 1.0 # float: 1.0 for similar, 0.0 for dissimilar
                  })
                
                flattened_pairs.append(v[0])
                flattened_pairs.append(v[1])
        
        
        def extract_pairs_with_min_diff(index_list, min_diff=5):
                x = index_list.copy()  # Avoid modifying input list
                pairs = []  # Store pairs as list of tuples
                
                while len(x) > 1:
                        random.shuffle(x)
                        if len(x) >= 2 and abs(x[0] - x[1]) >= min_diff:
                                a = x.pop(0)
                                b = x.pop(0)
                                pairs.append((a, b))
                        else:
                                found = False
                                for i in range(len(x)):
                                        for j in range(i + 1, len(x)):
                                                if abs(x[i] - x[j]) >= min_diff:
                                                        found = True
                                                        break
                                        if found:
                                                break
                                if not found:
                                        # Pair remaining elements
                                        while len(x) >= 2:
                                                a = x.pop(0)
                                                b = x.pop(0)
                                                pairs.append((a, b))
                
                return pairs, x  # Return pairs and remaining elements
        
        random.shuffle(flattened_pairs)
        random_index_pair_for_negative, x_rest = extract_pairs_with_min_diff (list(range(0, len(flattened_pairs))), 20)
        contrastive_negative_pairs = [{'anchor': flattened_pairs[i[0]], 'positive': flattened_pairs[i[1]], 'label': 0} for i in random_index_pair_for_negative]

        print(f"create contrastive positive pairs: {len(contrastive_positive_pairs)}")
        print(f"create contrastive negative pairs: {len(contrastive_negative_pairs)}")
        return contrastive_positive_pairs, contrastive_negative_pairs

 
    
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
        ordered_grouped_pairs = {group: all_pairs[group].copy() for group in sorted_groups if all_pairs[group]}
        
        print(f"Creating fixed-size batches maintaining hierarchy order")
        print(f"Initial groups: {list(ordered_grouped_pairs.keys())}")
        
        batches = []
        
        # Continue until no more complete batches can be formed
        while True:
            # Get available groups (those with chunks remaining)
            available_groups = {k: v for k, v in ordered_grouped_pairs.items() if v}
            
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
                if ordered_grouped_pairs[key]:
                    chunk = ordered_grouped_pairs[key].pop(0)
                    batch.append(chunk)
            
            # Only add batch if it's complete
            if len(batch) == batch_size:
                batches.append(batch)
                if len(batches) % 20 == 0:
                    remaining_groups = len([k for k, v in ordered_grouped_pairs.items() if v])
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
        

        all_pairs = self.create_sentence_level_pairs(chunks)
        
        # Step 3: Create fixed-size batches using cyclic sampling
        fixed_batches = self.create_fixed_size_batches(all_pairs, batch_size)
        
        # Convert to final MNRL format (remove metadata)
        mnrl_batches = []
        for batch in fixed_batches:
            mnrl_batch = []
            for pair in batch:
                mnrl_batch.append({
                    'anchor': pair[0],
                    'positive': pair[1]
                })
            mnrl_batches.append(mnrl_batch)
        
        final_pairs = sum(len(batch) for batch in mnrl_batches)
        print(f"✅ Fixed-size MNRL dataset: {len(mnrl_batches)} batches with {final_pairs} total pairs")
        print(f"Batch size consistency: All batches have exactly {batch_size} pairs")
        print("✅ Maximum diversity guaranteed within each batch")
        
        return mnrl_batches


if __name__ == "__main__":
    # Create dataset builder
    # Add this at the start of your script
    import torch
    torch.cuda.empty_cache()    
    builder = AdaptiveMNRLDatasetBuilder()
    
    # Load chunks from file 
    file_path = "data/data_processed/Lux_cssf18_698eng_chunked_blocks.json"
    # file_path ="data/data_processed/Basel_III_chunked_blocks.json"
    
    
    # Load chunks list
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks from {file_path}")

    # builder.create_sentence_level_pairs(chunks)
    a, b = builder.create_contrastive_positive_negative_pairs(chunks)


    
    # # Create fixed-size MNRL dataset with correct hierarchy (RECOMMENDED)
    # print("=== Creating Fixed-Size MNRL Dataset ===")
    # mnrl_batches = builder.generate_fixed_size_mnrl_batches(chunks, batch_size=8)
    
    # if mnrl_batches:
    #     # Save fixed-size MNRL dataset
    #     output_file = f"training_data/mnrl_fixed_batch_dataset_{Path(file_path).stem.replace('_chunked_blocks', '')}.json"
    #     with open(output_file, 'w', encoding='utf-8') as f:
    #         json.dump(mnrl_batches, f, indent=2, ensure_ascii=False)
    #     print(f"Fixed-size MNRL dataset saved to {output_file}")
        
    #     total_pairs = sum(len(batch) for batch in mnrl_batches)
    #     print(f"Generated {len(mnrl_batches)} batches with {total_pairs} total training pairs")
    #     print("✅ Dataset optimized for fixed-size batch MNRL training with correct hierarchy")
    # else:
    #     print("No dataset created - check your input chunks!")