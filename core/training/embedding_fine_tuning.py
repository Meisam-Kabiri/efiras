#!/usr/bin/env python3
"""
Fine-tune Sentence Transformers for Regulatory Document Embeddings
"""

import json
import logging
import random
# from torch.utils.data import DataLoader  # No need as we defiend owr own custom class
from typing import Dict, List, Tuple

import torch
from sentence_transformers import (InputExample, SentenceTransformer,
                                   evaluation, losses)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create custom DataLoader that preserves our diverse batch structure
# PyTorch's DataLoader expects: __iter__() method, __len__() method
# So we define a new class for out batch format
class DiverseBatchDataLoader:
    def __init__(self, batch_examples):
        self.batch_examples = batch_examples

    def __iter__(self):
        # Yield each diverse batch exactly as we created it
        for batch in self.batch_examples:
            yield batch

    def __len__(self):
        return len(self.batch_examples)

        # ADD THIS - sentence-transformers might expect it

    @property
    def batch_size(self):
        return len(self.batch_examples[0]) if self.batch_examples else 8


class RegulatoryEmbeddingTrainer:
    def __init__(
        self,
        base_model: str = "sentence-transformers/all-mpnet-base-v2",
        output_path: str = "./financial_embeddings_finetuned",
    ):

        self.base_model_name = base_model
        self.output_path = output_path
        self.model = SentenceTransformer(base_model)

        logger.info(f"Loaded base model: {base_model}")
        logger.info(f"Model device: {self.model.device}")

    def fine_tune_contrastive(
        self,
        contrastive_pairs: List[Dict],
        epochs: int = 5,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        batch_size: int = 16,
    ):
        """Fine-tune using Contrastive Loss"""
        from torch.utils.data import DataLoader

        logger.info(
            f"Creating contrastive training with {len(contrastive_pairs)} pairs"
        )

        # Create InputExamples with explicit labels
        train_examples = []
        for pair in contrastive_pairs:
            example = InputExample(
                texts=[pair["anchor"], pair["positive"]],
                label=float(pair["label"]),  # ContrastiveLoss needs float labels
            )
            train_examples.append(example)

        # Split train/validation
        random.shuffle(train_examples)
        split_point = int(0.8 * len(train_examples))
        train_data = train_examples[:split_point]
        val_data = train_examples[split_point:]

        # Create simple evaluator
        evaluator = (
            evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
                val_data, name="contrastive_eval"
            )
            if val_data
            else None
        )

        # Check balance
        positive_count = sum(1 for ex in train_data if ex.label == 1.0)
        negative_count = len(train_data) - positive_count
        logger.info(
            f"Training: {len(train_data)} pairs (Pos: {positive_count}, Neg: {negative_count})"
        )

        # Create standard DataLoader (no custom batching needed)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

        # Use Contrastive Loss instead of MNRL
        train_loss = losses.ContrastiveLoss(self.model)
        logger.info("Using Contrastive Loss with explicit positive/negative pairs")

        # Train (same as before, just different loss)
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            evaluator=evaluator,  # Skip evaluation
            warmup_steps=warmup_steps,
            output_path=self.output_path,
            save_best_model=True,
            optimizer_params={"lr": learning_rate},
            scheduler="WarmupLinear",
        )

        logger.info(
            f"✅ Contrastive fine-tuning complete! Model saved to: {self.output_path}"
        )

    def fine_tune_with_diverse_batches(
        self,
        diverse_batches: List[List[Dict]],
        epochs: int = 5,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
    ):
        """Fine-tune using diverse batches EXACTLY as created - no flattening!"""

        # Respect the batch structure we created
        batch_sizes = [len(batch) for batch in diverse_batches]
        total_examples = sum(batch_sizes)
        logger.info(
            f"Loaded {len(diverse_batches)} diverse batches with {total_examples} total examples"
        )
        logger.info(
            f"Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, avg={sum(batch_sizes)/len(batch_sizes):.1f}"
        )
        logger.info("✅ Using your exact batch structure - NO FLATTENING!")

        # Split batches (not examples) for train/validation 80/20 percent
        random.shuffle(diverse_batches)  # Shuffle all batches first
        split_point = int(0.8 * len(diverse_batches))
        train_batches = diverse_batches[:split_point]
        val_batches = diverse_batches[split_point:]

        logger.info(f"Training with {len(train_batches)} diverse batches")
        logger.info(f"Validation with {len(val_batches)} diverse batches")

        # Convert to InputExample format ONLY for sentence-transformers, preserving batch structure
        train_batch_examples = []
        for batch in train_batches:
            batch_inputs = []
            for item in batch:
                example = InputExample(texts=[item["anchor"], item["positive"]])
                batch_inputs.append(example)
            train_batch_examples.append(batch_inputs)

        # Create MNRL loss
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        logger.info("Using Multiple Negatives Ranking Loss with diverse batches")

        # Create the dataloader that respects our batch structure
        train_dataloader = DiverseBatchDataLoader(train_batch_examples)

        # Create simple validation evaluator from validation batches
        val_examples_flat = []
        for batch in val_batches:
            for item in batch:
                val_examples_flat.append(
                    InputExample(texts=[item["anchor"], item["positive"]])
                )

        evaluator = (
            evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
                val_examples_flat, name="diverse_batch_eval"
            )
            if val_examples_flat
            else None
        )

        # Training with our diverse batch structure
        logger.info("Starting diverse batch training...")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(
            f"Training batches: {len(train_batch_examples)} (variable sizes preserved)"
        )

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            evaluator=evaluator,
            evaluation_steps=100,
            # warmup_steps=warmup_steps,
            output_path=self.output_path,
            save_best_model=True,
            optimizer_params={"lr": learning_rate},
            # scheduler='WarmupLinear'
        )

        logger.info(
            f"✅ Diverse batch fine-tuning complete! Model saved to: {self.output_path}"
        )
        logger.info("Your batch diversity structure was preserved throughout training!")
