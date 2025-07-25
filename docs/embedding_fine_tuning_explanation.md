# Why Fine-tune Transformers for Sentence Similarity?

## The Question
Why do we need similarity/dissimilarity training for transformers? Aren't transformers just about attention within sentences?

## Understanding Different Transformer Types

### 1. **Generative Transformers** (GPT, LLaMA)
- **Purpose**: Generate text
- **Architecture**: Decoder-only, autoregressive
- **Training**: Next token prediction
- **Attention**: Within sequences for generation

### 2. **Embedding Transformers** (BERT, RoBERTa)
- **Purpose**: Create vector representations
- **Architecture**: Encoder-only, bidirectional
- **Training**: Masked language modeling + Next sentence prediction
- **Attention**: Bidirectional context understanding

### 3. **Sentence Transformers** (Sentence-BERT, E5)
- **Purpose**: Compare and rank text chunks
- **Architecture**: BERT-based with pooling layers
- **Training**: Contrastive learning on sentence pairs
- **Attention**: Optimized for semantic similarity

## The Problem with Base Transformers

Base BERT/transformers are trained on:
- **Masked Language Modeling**: Fill in blanks (`The [MASK] is red`)
- **Next Sentence Prediction**: Binary classification (does sentence B follow A?)

But they're **terrible** at understanding semantic similarity between arbitrary text chunks.

### Example Problem in RAG Systems
```python
# When user asks: "What are Basel III capital requirements?"
# RAG system needs to find MOST SIMILAR chunks from database:

chunks = [
    "Basel III establishes minimum capital ratios for banks",  # VERY RELEVANT ✅
    "Banks must maintain adequate liquidity buffers",          # SOMEWHAT RELEVANT ⚠️
    "Customer complaints procedures are detailed in...",       # NOT RELEVANT ❌
]
```

**Base transformers can't reliably rank these by similarity!**

## Difference Between Embedding Transformers vs Sentence Transformers

| Concept | Embedding Transformers (e.g., BERT) | Sentence Transformers (e.g., SBERT) |
|---------|-------------------------------------|-------------------------------------|
| **Purpose** | Output contextual embeddings for each token | Output a single dense vector for a full sentence |
| **Output** | Sequence of token vectors (one per word) | One fixed-size embedding per sentence |
| **Use Case** | Token classification (NER, QA, etc.) | Sentence similarity, clustering, semantic search |
| **Architecture** | Vanilla BERT | Siamese / triplet networks using BERT as backbone |
| **Training** | Trained for token-level tasks (MLM, NSP) | Fine-tuned for sentence-level tasks (e.g., semantic similarity) |
| **Embedding** | Not directly usable for sentence similarity unless pooled | Directly optimized for sentence embeddings |
| **Example Model** | `bert-base-uncased` | `all-MiniLM-L6-v2`, `paraphrase-MPNet-base-v2` |

### 🔹 Illustration:
Suppose we have the sentence: **"Paris is the capital of France."**

**BERT**: gives 768-dimensional embedding for each token like `[CLS]`, `"Paris"`, `"is"`, ..., `"France"`, `[SEP]`.

**Sentence Transformer**: gives one 768-dimensional vector that represents the whole sentence meaning.

### Visual Representation:
```
Input: "Paris is the capital of France."

BERT Output:
[CLS]   → [0.1, 0.2, 0.3, ..., 0.7]  # 768 dimensions
Paris   → [0.4, 0.1, 0.8, ..., 0.2]  # 768 dimensions  
is      → [0.2, 0.9, 0.1, ..., 0.4]  # 768 dimensions
the     → [0.7, 0.3, 0.5, ..., 0.9]  # 768 dimensions
capital → [0.1, 0.8, 0.2, ..., 0.3]  # 768 dimensions
of      → [0.5, 0.4, 0.9, ..., 0.1]  # 768 dimensions
France  → [0.8, 0.1, 0.4, ..., 0.6]  # 768 dimensions
[SEP]   → [0.3, 0.7, 0.1, ..., 0.8]  # 768 dimensions

Sentence Transformer Output:
Sentence → [0.4, 0.2, 0.7, ..., 0.5]  # 768 dimensions (single vector)
```

## Training Methods for Sentence Transformers

**Contrastive Learning** is the overarching concept of learning by contrasting similar vs dissimilar examples. It includes several specific methods:

### 1. **Siamese Networks with Contrastive Loss**
```python
# Train model to learn embeddings:
similar_pairs = [
    ("Basel III capital requirements", "Banks must maintain minimum capital ratios", 1),
    ("Liquidity coverage ratio", "LCR calculation methodology", 1)
]

dissimilar_pairs = [
    ("Basel III capital requirements", "Customer complaint procedures", 0),
    ("Risk management framework", "Office holiday schedule", 0)
]
```

**How it works**:
- Two identical neural networks (Siamese) process sentence pairs
- Contrastive loss pulls similar pairs closer, pushes dissimilar pairs apart
- Outputs dense embeddings for vector search (good for RAG)

### 2. **Multiple Negatives Ranking Loss (RECOMMENDED FOR RAG)**
```python
# MNRL Training Data Format (NO EXPLICIT NEGATIVES!)
{"anchor": "Basel III capital requirements", "positive": "Banks must maintain minimum capital ratios"}
{"anchor": "Risk management framework", "positive": "Risk assessment procedures"}
{"anchor": "Liquidity coverage ratio", "positive": "LCR calculation methodology"}
```

**Why MNRL is Superior for RAG Systems:**
- **Optimized for Retrieval**: Specifically designed for dense retrieval tasks
- **Efficient Training**: Uses in-batch negatives (no manual negative sampling needed)
- **Better Ranking**: Learns to rank multiple candidates, not just binary similarity
- **State-of-the-art**: Used by modern embedding models (E5, BGE, etc.)
- **Regulatory Structure**: Can leverage document hierarchy automatically

**CRITICAL DIFFERENCE: No Explicit Negatives Required**

**Traditional Contrastive Learning Dataset:**
```python
{"sentence1": "Basel III capital", "sentence2": "Capital requirements", "label": 1}  # Similar
{"sentence1": "Basel III capital", "sentence2": "Holiday schedule", "label": 0}     # Dissimilar
```
❌ **Requires manual negative selection** - doubles dataset size

**MNRL Dataset:**
```python
{"anchor": "Basel III capital", "positive": "Capital requirements"}
{"anchor": "Risk management", "positive": "Risk assessment"}
```
✅ **No explicit negatives needed** - half the dataset size!

**How MNRL Creates Negatives During Training:**

**Training Batch (size=4):**
```python
Batch = [
  ("Basel III capital", "Capital requirements"),           # Target pair
  ("Risk management", "Risk assessment procedures"),      # Negative 1  
  ("Liquidity ratios", "LCR calculation"),               # Negative 2
  ("Operational risk", "Risk measurement methods")        # Negative 3
]
```

**For anchor "Basel III capital":**
- **Positive**: "Capital requirements" (target)
- **In-batch Negatives**: "Risk assessment procedures", "LCR calculation", "Risk measurement methods"

**MNRL Training Logic:**
1. **Pull anchor closer** to its positive
2. **Push anchor away** from all other positives in the batch
3. **Automatic hard negatives** - other regulatory content is harder to distinguish than random text
4. **Dynamic negatives** - different negatives every batch

**Benefits of In-Batch Negatives:**
- **No manual curation** of negative examples
- **Regulatory domain negatives** - harder than random text  
- **Scales with batch size** - larger batches = more negatives
- **Balanced training** - always equal positives/negatives
- **Computational efficiency** - reuses batch embeddings

**For Regulatory Documents:**
- **Anchor**: A chunk about specific regulation
- **Positive**: Another chunk from same section/topic  
- **Negatives**: Other chunks in batch (automatic during training)

### 3. **Triplet Loss**
```python
# Anchor, Positive, Negative
("Basel III requirements", "Capital adequacy rules", "Parking regulations")
# Model learns: anchor closer to positive than negative
```



### 4. **Classification Fine-tuning** (What EFIRAS code structure suggests)
```python
# Train a classifier to predict similarity
model(sentence1, sentence2) → probability score (0-1)
# Use case: "Are these sentences similar?"
```

⚠️ **Note**: For RAG systems, **embedding-based approaches** (methods 1-3) are preferred over classification because they enable efficient vector search in databases like FAISS/Pinecone.

### 5. **Natural Language Inference (NLI)**
```python
# Entailment, Contradiction, Neutral
("All banks must maintain capital ratios", "Banks need adequate capital", "ENTAILMENT")
("High capital ratios", "Low capital ratios", "CONTRADICTION")
```

## Other Fine-tuning Approaches

### 1. **Domain-Specific Pretraining**
```python
# Continue pretraining on regulatory documents
model.train_on_domain_text([
    "basel_documents.txt",
    "mifid_regulations.txt", 
    "cssf_circulars.txt"
])
```

**What it is**: Continue training base models (BERT) on domain text using original objectives (MLM, NSP) to learn domain vocabulary and patterns.

**Why EFIRAS doesn't use it**:
- **Different Goal**: Learns domain vocabulary vs. sentence similarity
- **Resource Intensive**: Needs millions of tokens, days of training
- **Two-Stage Process**: Still needs similarity fine-tuning afterward
- **EFIRAS is Task-Specific**: Direct optimization for similarity matching

| Aspect | Domain Pretraining | EFIRAS Approach |
|--------|-------------------|-----------------|
| **Data Need** | Millions of tokens (raw text) | Thousands of pairs (structured) |
| **Training Time** | Days/weeks | Hours |
| **Goal** | Learn domain vocabulary | Learn similarity matching |
| **Resources** | High GPU, massive corpus | Moderate GPU, document pairs |
| **Use Case** | Multiple NLP tasks | Specific RAG similarity |

### 2. **Question-Answer Pairs**
```python
qa_pairs = [
    ("What is Tier 1 capital?", "Tier 1 capital includes common equity..."),
    ("How is LCR calculated?", "LCR is calculated as high-quality liquid assets...")
]
```

### 3. **Synthetic Data Generation**
```python
# Use GPT to generate similar/dissimilar pairs
prompt = "Generate 5 similar sentences to: 'Basel III capital requirements'"
```

### 4. **Multi-task Learning**
```python
# Train on multiple objectives simultaneously
tasks = [
    "similarity_pairs",
    "question_answering", 
    "classification",
    "named_entity_recognition"
]
```

## Why EFIRAS Approach Makes Sense

### For Regulatory Documents:
- **High Precision Needed**: Wrong similarity = wrong legal interpretation
- **Domain-Specific**: Generic models don't understand "Tier 1 capital" relationships
- **Hierarchical Structure**: Code exploits document structure (headers, sections)
- **Regulatory Nuances**: Understanding concepts like "capital adequacy" vs "liquidity coverage"

### EFIRAS Dataset Creation Strategy:
```python
# Similar pairs from:
- Adjacent blocks (sequential context)
- Same regulation type (Basel III internal consistency)
- Shared financial concepts (cross-regulation similarity)

# Dissimilar pairs from:
- Different regulation types
- Different document sections
- Concept mismatch validation
```

## Alternative: Why Not Just Use OpenAI Embeddings?

```python
# Generic embeddings might fail:
openai.embeddings("Basel III capital requirements")  
# ↳ Generic understanding of "capital" and "requirements"

vs

your_finetuned_model("Basel III capital requirements")  
# ↳ Regulatory-specific understanding:
#   - Basel III as specific regulation
#   - Capital as banking concept
#   - Requirements as regulatory obligations
```

### Domain-Specific Advantages:
- **Regulatory Terminology**: Understands "Tier 1", "LCR", "CET1"
- **Hierarchical Relationships**: Knows Basel III → Capital → Tier 1 → Common Equity
- **Contextual Nuances**: Distinguishes "capital" (banking) vs "capital" (general)

## The Training Process

### 1. **Data Preparation**
```python
# From document structure
blocks = load_processed_blocks("basel_iii.json")

# Create training pairs
similar_pairs = generate_similar_pairs(blocks)
dissimilar_pairs = generate_dissimilar_pairs(blocks)
```

### 2. **Model Architecture**
```python
# Sentence Transformer pipeline
base_model = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(base_model)

# Add pooling for sentence representations
model.add_pooling_layer()
```

### 3. **Training Loop**
```python
# Contrastive loss
for batch in dataloader:
    embeddings1 = model.encode(batch['sentence1'])
    embeddings2 = model.encode(batch['sentence2'])
    
    loss = contrastive_loss(embeddings1, embeddings2, batch['labels'])
    loss.backward()
    optimizer.step()
```

## Real-World Impact

### Before Fine-tuning:
```python
query = "Basel III capital requirements"
results = [
    "Banking regulations overview",      # Score: 0.7
    "Capital market analysis",          # Score: 0.8 (Wrong!)
    "Basel III minimum capital ratios"  # Score: 0.6 (Should be highest!)
]
```

### After Fine-tuning:
```python
query = "Basel III capital requirements"
results = [
    "Basel III minimum capital ratios",  # Score: 0.95 ✅
    "Banking regulations overview",       # Score: 0.7
    "Capital market analysis",           # Score: 0.3 ✅
]
```

## Conclusion

Fine-tuning transformers for sentence similarity is about creating **domain-expert embeddings** that understand:
- **Semantic relationships** specific to your domain
- **Hierarchical structures** in your documents
- **Contextual nuances** that generic models miss

For regulatory documents, this means the difference between finding the right legal provision and potentially missing critical compliance information.

**Bottom line**: You're training the model to be a regulatory document expert, not just a generic text processor.