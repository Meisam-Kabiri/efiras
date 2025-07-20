# MNRL Dataset Builder Documentation

## Overview

The `AdaptiveMNRLDatasetBuilder` creates training datasets for embedding fine-tuning using **Multiple Negatives Ranking Loss (MNRL)**. It processes regulatory documents and generates positive pairs for contrastive learning.

## Core Algorithm

### 1. Hierarchy-Based Grouping

Groups document chunks by their deepest hierarchy level:

```
"Part I > Chapter 2 > Section 3" → "Section 3" group
```

- **Single group membership**: Each chunk belongs to exactly one group
- **Deepest level only**: Uses the most specific document section

### 2. Positive Pair Generation

Creates similar content pairs within the same hierarchy group:

- **Within-group pairing**: Pairs chunks only from the same group
- **Adjacent pairing**: Links nearby chunks for semantic similarity
- **Text truncation**: Limits anchor/positive text to 500 characters

### 3. Fixed-Size Batch Creation

Uses round-robin sampling to ensure training diversity:

```python
# Batch creation process:
1. Group chunks by deepest hierarchy level
2. Create pairs within each group  
3. Use round-robin sampling across groups
4. Fill batches with one pair from each different group
5. Cycle through groups until batch_size reached
```

## Input/Output

### Input Structure
```json
[
  {
    "text": "Capital requirements must be maintained...",
    "enriched_headers": "Part I > Chapter 1 > Section 1.1",
    "page_number": 15,
    "chunk_id": "chunk_001"
  }
]
```

### Output Structure
```json
[
  [
    {
      "anchor": "Capital requirements must be maintained...",
      "positive": "Banks shall calculate their capital ratios..."
    },
    {
      "anchor": "Liquidity risk management involves...",
      "positive": "Daily liquidity reporting must include..."
    }
  ]
]
```

## Processing Example

Given document structure:
```
Part I
├── Chapter 1
│   ├── Section 1.1 (2 chunks)
│   └── Section 1.2 (2 chunks)
└── Chapter 2
    └── Section 2.1 (2 chunks)
```

### Step-by-Step Process:

1. **Group by deepest level**:
   - `Section 1.1`: [chunk1, chunk2]
   - `Section 1.2`: [chunk3, chunk4] 
   - `Section 2.1`: [chunk5, chunk6]

2. **Create pairs within groups**:
   - Section 1.1: 1 pair
   - Section 1.2: 1 pair
   - Section 2.1: 1 pair

3. **Round-robin batching** (batch_size=3):
   ```
   Batch 1: [Section_1.1_pair, Section_1.2_pair, Section_2.1_pair]
   ```

## Key Features

- **Maximum diversity**: Each batch contains pairs from different document sections
- **Consistent batch sizes**: All batches have exactly the same number of pairs
- **Semantic similarity**: Pairs within groups share similar content
- **Training ready**: Direct input for MNRL loss function

## Usage

```python
# Load chunks from file
with open('document_chunks.json', 'r') as f:
    chunks = json.load(f)

# Create dataset
builder = AdaptiveMNRLDatasetBuilder()
batches = builder.generate_mnrl_batches(chunks, batch_size=8)

# Save results
with open('training_data.json', 'w') as f:
    json.dump(batches, f)
```

## Configuration

- **batch_size**: Number of pairs per batch (default: 8)
- **text_limit**: Maximum characters per anchor/positive (500)
- **min_chunk_length**: Minimum chunk size to process (30 characters)

## Training Benefits

- **Diverse positive examples** within each batch
- **Consistent batch sizes** for stable gradient updates
- **Hierarchical learning** from document structure
- **Balanced representation** across document sections