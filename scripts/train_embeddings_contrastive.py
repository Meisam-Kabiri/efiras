#!/usr/bin/env python3
"""
EFIRAS Contrastive Embedding Training Runner
"""

import json
import os
import sys
from pathlib import Path

import torch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))


def main():
    """Main contrastive training pipeline"""

    print("EFIRAS Contrastive Embedding Training")
    print("=" * 40)

    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ Using CPU (slower)")

    # Load data
    data_file = "data/data_processed/Lux_cssf18_698eng_chunked_blocks.json"
    data_file2 = "data/data_processed/Basel_III_chunked_blocks.json"
    print(f"\n📂 Loading: {data_file}")

    with open(data_file, "r") as f:
        chunks = json.load(f)

    with open(data_file2, "r") as f1:
        chunks2 = json.load(f1)

    # Create contrastive pairs
    print("🔄 Creating contrastive pairs...")
    from training.embedding_dataset_preparation import \
        AdaptiveMNRLDatasetBuilder

    builder = AdaptiveMNRLDatasetBuilder()
    positive_pairs, negative_pairs = builder.create_contrastive_positive_negative_pairs(
        chunks
    )
    positive_pairs2, negative_pairs2 = (
        builder.create_contrastive_positive_negative_pairs(chunks2)
    )
    all_pairs = positive_pairs + negative_pairs
    all_pairs += positive_pairs2 + negative_pairs2

    print(
        f"✅ Created {len(positive_pairs)} positive + {len(negative_pairs)} negative pairs"
    )

    # Train model
    print("\n🚀 Training with contrastive loss...")
    from training.embedding_fine_tuning import RegulatoryEmbeddingTrainer

    trainer = RegulatoryEmbeddingTrainer(
        base_model="sentence-transformers/all-mpnet-base-v2",
        output_path="models/efiras_contrastive_embeddings",
    )

    trainer.fine_tune_contrastive(
        contrastive_pairs=all_pairs, epochs=3, learning_rate=2e-5, batch_size=16
    )

    print("✅ Training complete! Model saved to: models/efiras_contrastive_embeddings")


if __name__ == "__main__":
    main()
