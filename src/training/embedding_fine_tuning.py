#!/usr/bin/env python3
"""
Fine-tune Sentence Transformers for Regulatory Document Embeddings
"""

import json
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
# from torch.utils.data import DataLoader  # No need as we defiend owr own custom class 
from typing import List, Dict, Tuple
import logging

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
            

class RegulatoryEmbeddingTrainer:
    def __init__(self, 
                 base_model: str = "sentence-transformers/all-mpnet-base-v2",
                 output_path: str = "./financial_embeddings_finetuned"):
        
        self.base_model_name = base_model
        self.output_path = output_path
        self.model = SentenceTransformer(base_model)
        
        logger.info(f"Loaded base model: {base_model}")
        logger.info(f"Model device: {self.model.device}")
    
    
    def fine_tune_with_diverse_batches(self, 
                                      diverse_batches: List[List[Dict]],
                                      epochs: int = 5,
                                      learning_rate: float = 2e-5,
                                      warmup_steps: int = 100):
        """Fine-tune using diverse batches EXACTLY as created - no flattening!"""
        
        # Respect the batch structure we created
        batch_sizes = [len(batch) for batch in diverse_batches]
        total_examples = sum(batch_sizes)
        logger.info(f"Loaded {len(diverse_batches)} diverse batches with {total_examples} total examples")
        logger.info(f"Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, avg={sum(batch_sizes)/len(batch_sizes):.1f}")
        logger.info("✅ Using your exact batch structure - NO FLATTENING!")
        
        # Split batches (not examples) for train/validation 80/20 percent
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
                example = InputExample(texts=[item['anchor'], item['positive']])
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
                val_examples_flat.append(InputExample(texts=[item['anchor'], item['positive']]))
        
        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
            val_examples_flat, name="diverse_batch_eval"
        ) if val_examples_flat else None
        
        # Training with our diverse batch structure
        logger.info("Starting diverse batch training...")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Training batches: {len(train_batch_examples)} (variable sizes preserved)")
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            evaluator=evaluator,
            evaluation_steps=100,
            warmup_steps=warmup_steps,
            output_path=self.output_path,
            save_best_model=True,
            optimizer_params={'lr': learning_rate},
            scheduler='WarmupLinear'
        )
        
        logger.info(f"✅ Diverse batch fine-tuning complete! Model saved to: {self.output_path}")
        logger.info("Your batch diversity structure was preserved throughout training!")