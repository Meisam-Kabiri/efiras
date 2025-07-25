# Training Performance Troubleshooting Guide

## Issue: Performance Degradation with Multi-Document Training

### Problem Analysis
- **NaN evaluation metrics** across all epochs indicate training failure
- **Mixed domains** (Basel III banking + Luxembourg fund regulations) create conflicting learning signals
- **No valid evaluation pairs** prevent proper progress monitoring

### Immediate Fixes

#### 1. Train on Single Document First
```bash
# Edit train_embeddings.py - use only one document
processed_files = [
    "data_processed/Lux_cssf18_698eng_chunked_blocks.json"  # Start with this
    # "data_processed/Basel_III_chunked_blocks.json",      # Add later
]
```

#### 2. Fix Evaluation Data Generation
Add evaluation dataset creation in `embedding_dataset_preparation.py`:

```python
def create_evaluation_pairs(self, chunks: List[Dict], num_pairs: int = 100) -> List[Tuple]:
    """Create evaluation pairs for monitoring training progress"""
    positive_pairs = []
    negative_pairs = []
    
    # Create positive pairs (same section/topic)
    level_groups = self.group_by_lowest_level(chunks)
    for level, level_chunks in level_groups.items():
        if len(level_chunks) >= 2:
            pairs = list(combinations(level_chunks, 2))
            positive_pairs.extend(random.sample(pairs, min(len(pairs), num_pairs//4)))
    
    # Create negative pairs (different sections)
    level_list = list(level_groups.keys())
    for _ in range(num_pairs//2):
        if len(level_list) >= 2:
            level1, level2 = random.sample(level_list, 2)
            chunk1 = random.choice(level_groups[level1])
            chunk2 = random.choice(level_groups[level2])
            negative_pairs.append((chunk1, chunk2, 0))  # 0 = dissimilar
    
    # Add positive labels
    positive_pairs = [(p[0], p[1], 1) for p in positive_pairs]  # 1 = similar
    
    return positive_pairs + negative_pairs
```

#### 3. Reduce Learning Rate
```python
config = {
    "learning_rate": 5e-6,  # Much lower for stability
    "epochs": 2,            # Fewer epochs initially
    "batch_size": 4,        # Smaller batches
}
```

#### 4. Add Evaluation Step
Modify the fine-tuning to include evaluation:

```python
# In fine_tune_with_diverse_batches()
eval_pairs = self.create_evaluation_pairs(chunks)
evaluator = evaluation.EmbeddingSimilarityEvaluator(
    sentences1=[p[0]['text'] for p in eval_pairs],
    sentences2=[p[1]['text'] for p in eval_pairs], 
    scores=[p[2] for p in eval_pairs],
    name="regulatory_similarity"
)
```

### Long-term Strategy

#### Progressive Training Approach
1. **Stage 1**: Train on Luxembourg docs only (homogeneous domain)
2. **Stage 2**: Fine-tune the trained model with Basel III (domain adaptation)
3. **Stage 3**: Joint training with balanced sampling

#### Document-Specific Training
Create separate models for different regulatory domains:
- `models/lux_fund_embeddings/` for Luxembourg regulations
- `models/basel_banking_embeddings/` for Basel III
- `models/unified_financial_embeddings/` for combined (if needed)

### Quick Test Commands

```bash
# 1. Clean training - single document
python train_embeddings.py  # Choose option 1, then 2

# 2. Test model quality
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('models/efiras_financial_embeddings')
embeddings = model.encode(['capital requirements', 'liquidity ratios'])
print('Model loaded successfully!')
"

# 3. Compare with baseline
python compare_models_advanced.py
```

### Expected Results After Fix
- Evaluation metrics should show actual numbers (not NaN)
- Cosine similarity scores between 0.3-0.8 for good performance
- Training loss should decrease consistently

### Monitoring Training Health
```python
# Add to training loop
if epoch % 1 == 0:
    print(f"Epoch {epoch}: Loss = {train_loss:.4f}")
    if hasattr(evaluator, 'latest_score'):
        print(f"Evaluation Score: {evaluator.latest_score:.4f}")
```