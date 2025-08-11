#!/usr/bin/env python3
"""
EFIRAS Embedding Training Runner
Entry point for training custom embeddings
"""

import sys
import os
from pathlib import Path
import json
import torch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))
batch_size = 8

def check_gpu_availability():
    """Check and display GPU availability"""
    print("\n🔍 GPU Availability Check:")
    print("=" * 30)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        
        device_props = torch.cuda.get_device_properties(current_device)
        
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        print(f"🔢 GPU Count: {gpu_count}")
        print(f"📱 Current Device: {current_device}")
        print(f"🏷️  GPU Name: {gpu_name}")
        print(f"💾 GPU Memory: {device_props.total_memory / 1024**3:.1f} GB")
        print(f"⚡ CUDA Cores: {device_props.multi_processor_count * 128}")  # Approximate for RTX series
        print(f"🔧 Compute Capability: {device_props.major}.{device_props.minor}")
        return True
    else:
        print("❌ CUDA not available - training will use CPU")
        print("💡 For faster training, ensure CUDA-enabled PyTorch is installed")
        return False

def main():
    """Main training pipeline runner"""
    
    print("EFIRAS Embedding Training Pipeline")
    print("=" * 50)
    
    # Check GPU availability first
    check_gpu_availability()
    
    choice = input("""
                    Choose training pipeline:
                    1. Create fixed-size MNRL dataset with correct hierarchy (RECOMMENDED for RAG)
                    2. Fine-tune embedding model with batch-aware training
                    3. Full pipeline (1 + 2)

                    Enter choice (1-3): """).strip()
    
    if choice in ["1", "3"]:
        print("\n🎯 Creating fixed-size MNRL embedding dataset with correct hierarchy...")
        from training.embedding_dataset_preparation import AdaptiveMNRLDatasetBuilder
        
        # Available processed files
        processed_files = [
            # "data/data_processed/Basel_III_chunked_blocks.json",
            "data/data_processed/Lux_cssf18_698eng_chunked_blocks.json"
        ]
        
        for file_path in processed_files:
            if Path(file_path).exists():
                print(f"Processing: {file_path}")
                
                with open(file_path, 'r') as f:
                    chunks = json.load(f)
                builder = AdaptiveMNRLDatasetBuilder()
                # Use the new fixed-size algorithm with proper hierarchy detection
                dataset = builder.generate_fixed_size_mnrl_batches(chunks, batch_size=batch_size)
                
                if dataset:
                    # Save to training_data directory
                    doc_name = Path(file_path).stem.replace("_chunked_blocks", "")
                    output_path = f"training_data/mnrl_fixed_batch_dataset_{doc_name}.json"
                    
                    # Ensure directory exists
                    Path("training_data").mkdir(exist_ok=True)
                    
            
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(dataset, f, indent=2, ensure_ascii=False)
                    
                    print(f"✅ MNRL Dataset saved: {output_path}")
                    total_pairs = sum(len(batch) for batch in dataset)
                    print(f"Generated {len(dataset)} diverse batches with {total_pairs} total pairs")
                break
    
    if choice in ["2", "3"]:
        print("\n🚀 Fine-tuning embedding model...")
        from training.embedding_fine_tuning import RegulatoryEmbeddingTrainer
        
        # Check for dataset (prefer fixed-batch MNRL, fallback to old MNRL, then traditional)
        fixed_batch_files = list(Path("training_data").glob("*.json"))

        if fixed_batch_files:
            print(f"Found {len(fixed_batch_files)} dataset files:")
            for file in fixed_batch_files:
                print(f"  - {file}")
            
            # Load and combine all datasets
            all_batches = []
            for file_path in fixed_batch_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                    all_batches.extend(dataset)  # Add all batches from this file
                    print(f"Loaded {len(dataset)} batches from {file_path}")
            
            print(f"Combined total: {len(all_batches)} batches")


        
        # Verify this is our batch format
        if not isinstance(all_batches[0], list):
            raise ValueError("Expected diverse batch format (List[List[Dict]]) - use fine_tune() for flat datasets")
        
        # Configure training
        # Consider domain-specific models like sentence-transformers/msmarco-bert-base-dot-v5
        config = {
                "base_model": "sentence-transformers/all-mpnet-base-v2", # this is better without fine tunnig
                # "base_model": "intfloat/e5-base-v2", # This is better with fine-tunning
                "diverse_batches": all_batches,
                "output_path": "models/efiras_financial_embeddings",
                "epochs": 3,           # Start smaller, can increase
                "batch_size": batch_size,       # Better for MNRL
                "learning_rate": 1e-5,
                "warmup_steps": 100,   # Add warmup
                "evaluation_steps": 50 # Add evaluation
                }
        
        # Train model
        trainer = RegulatoryEmbeddingTrainer(
            base_model=config["base_model"],
            output_path=config["output_path"],
        )
        
            
        print("🎯 Using diverse batch training (preserves your batch structure)")
        trainer.fine_tune_with_diverse_batches(
            diverse_batches=config["diverse_batches"],
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            warmup_steps= config['warmup_steps']
        )
        
        print(f"✅ Model trained and saved: {config['output_path']}")
    
    print("\n🎉 Training pipeline completed!")

if __name__ == "__main__":
    main()