#!/usr/bin/env python3
"""
Compare trained model vs base model
"""

from sentence_transformers import SentenceTransformer

def compare_models():
    # Load both models
    base_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    trained_model = SentenceTransformer("models/efiras_financial_embeddings")
    
    # Test sentences
    regulatory_pair = [
        "Basel III capital adequacy requirements",
        "Banks must maintain minimum capital ratios"
    ]
    
    unrelated_pair = [
        "Basel III capital adequacy requirements", 
        "Employee vacation scheduling policies"
    ]
    
    print("🆚 Comparing Base vs MNRL-Trained Model")
    print("=" * 50)
    
    # Test regulatory similarity
    print("📊 Regulatory Similar Pair:")
    print(f"Text 1: {regulatory_pair[0]}")
    print(f"Text 2: {regulatory_pair[1]}")
    
    base_sim = base_model.similarity(
        base_model.encode([regulatory_pair[0]]), 
        base_model.encode([regulatory_pair[1]])
    )[0][0].item()
    
    trained_sim = trained_model.similarity(
        trained_model.encode([regulatory_pair[0]]), 
        trained_model.encode([regulatory_pair[1]])
    )[0][0].item()
    
    print(f"Base model similarity:    {base_sim:.3f}")
    print(f"Trained model similarity: {trained_sim:.3f}")
    print(f"Improvement: {trained_sim - base_sim:+.3f}")
    
    # Test unrelated similarity
    print("\n📊 Unrelated Pair:")
    print(f"Text 1: {unrelated_pair[0]}")
    print(f"Text 2: {unrelated_pair[1]}")
    
    base_sim_unrelated = base_model.similarity(
        base_model.encode([unrelated_pair[0]]), 
        base_model.encode([unrelated_pair[1]])
    )[0][0].item()
    
    trained_sim_unrelated = trained_model.similarity(
        trained_model.encode([unrelated_pair[0]]), 
        trained_model.encode([unrelated_pair[1]])
    )[0][0].item()
    
    print(f"Base model similarity:    {base_sim_unrelated:.3f}")
    print(f"Trained model similarity: {trained_sim_unrelated:.3f}")
    print(f"Change: {trained_sim_unrelated - base_sim_unrelated:+.3f}")
    
    print("\n🎯 Expected Results:")
    print("✅ Higher similarity for regulatory pairs")
    print("✅ Lower similarity for unrelated pairs")
    print("✅ Better distinction between related/unrelated")

if __name__ == "__main__":
    compare_models()