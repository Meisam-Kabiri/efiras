# EFIRAS Embedding Training

This directory contains the embedding training pipeline for EFIRAS.

## Directory Structure

```
src/training/
├── __init__.py
├── embedding_fine_tuning.py          # Main fine-tuning script
├── embedding_dataset_preparation.py  # Dataset creation from processed blocks
└── README.md
```

## Usage

### From Root Directory

```bash
# Run the interactive training pipeline
python train_embeddings.py

# Or run individual components
python -m src.training.embedding_dataset_preparation
python -m src.training.embedding_fine_tuning
```

### From Training Directory

```bash
cd src/training
python embedding_fine_tuning.py
```

## Data Flow

1. **Input**: Processed blocks from `data_processed/`
2. **Dataset Creation**: Creates training pairs in `training_data/`
3. **Fine-tuning**: Trains model and saves to `models/`
4. **Output**: Fine-tuned model ready for RAG system

## Configuration

Edit `train_embeddings.py` to modify:
- Base model (`all-mpnet-base-v2`)
- Training epochs (5)
- Batch size (16)
- Learning rate (2e-5)