#!/usr/bin/env python3
"""
Advanced Model Comparison Script for Luxembourg Regulatory Documents
Compare original vs fine-tuned embeddings on tricky questions
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

class EmbeddingModelComparator:
    def __init__(self):
        # Load original model
        print("Loading original sentence transformer model...")
        self.original_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        
        # Try to load fine-tuned model
        fine_tuned_path = "models/efiras_financial_embeddings"
        if Path(fine_tuned_path).exists():
            print(f"Loading fine-tuned model from {fine_tuned_path}...")
            self.fine_tuned_model = SentenceTransformer(fine_tuned_path)
        else:
            print(f"⚠️  Fine-tuned model not found at {fine_tuned_path}")
            print("Please run training first or specify correct path")
            self.fine_tuned_model = None
        
        # Load processed chunks
        self.chunks = self.load_chunks()
        
        # Cache for embeddings
        self.chunk_embeddings_cache = {}
        self.cache_dir = Path("data_processed/embeddings_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def load_chunks(self) -> List[Dict]:
        """Load processed chunks from Luxembourg document"""
        chunk_file = "data_processed/Lux_cssf18_698eng_chunked_blocks.json"
        # chunk_file = "data_processed/Lux_cssf18_698eng_processed_blocks.json"
        
        if not Path(chunk_file).exists():
            print(f"❌ Chunk file not found: {chunk_file}")
            return []
        
        with open(chunk_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # chunks = data.get('blocks', [])
        chunks = data
        print(f"Loaded {len(chunks)} chunks from Luxembourg regulatory document")
        return chunks
    
    def get_tricky_questions(self) -> List[Dict[str, str]]:
        """Generate tricky questions about Luxembourg regulatory requirements"""
        return [
            {
                "question": "What are the specific conditions for performing multiple mandates by governing body members in Luxembourg IFMs?",
                "category": "Governance",
                "difficulty": "High",
                "expected_sections": ["Section 4.1.3", "governing body", "multiple mandates"]
            },
            {
                "question": "How should an IFM handle conflicts of interest when managing both UCIs and individual portfolios?",
                "category": "Conflicts of Interest", 
                "difficulty": "Very High",
                "expected_sections": ["Sub-section 5.5.7", "conflicts of interest", "portfolio management"]
            },
            {
                "question": "What are the technical infrastructure requirements for business continuity in Luxembourg IFMs?",
                "category": "Technical Requirements",
                "difficulty": "High", 
                "expected_sections": ["Section 5.1.2", "business continuity", "technical infrastructure"]
            },
            {
                "question": "Under what circumstances can an IFM delegate the valuation function and what conditions apply?",
                "category": "Delegation",
                "difficulty": "Very High",
                "expected_sections": ["Section 6.6", "valuation function", "delegation"]
            },
            {
                "question": "What specific reporting obligations does an IFM have regarding risk management procedures to the CSSF?",
                "category": "Risk Management",
                "difficulty": "High",
                "expected_sections": ["Sub-section 5.3.1.5", "risk management procedure", "CSSF", "reporting"]
            },
            {
                "question": "How does the principle of proportionality apply to smaller IFMs regarding compliance requirements?",
                "category": "Proportionality",
                "difficulty": "Very High", 
                "expected_sections": ["Part V", "proportionality", "compliance"]
            },
            {
                "question": "What are the specific obligations for IFMs regarding EMIR compliance and derivative monitoring?",
                "category": "EMIR Compliance",
                "difficulty": "High",
                "expected_sections": ["Section 5.5.11", "EMIR", "derivatives", "monitoring"]
            },
            {
                "question": "When establishing a branch in another EU country, what notification procedures must Luxembourg IFMs follow?",
                "category": "Cross-border Operations", 
                "difficulty": "Very High",
                "expected_sections": ["Sub-chapter 1.1", "branch", "notification", "EU"]
            }
        ]
    
    def find_relevant_chunks(self, query: str, model: SentenceTransformer, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Find most relevant chunks using specified model with persistent caching"""
        if not self.chunks:
            return []
        
        # Get query embedding
        query_embedding = model.encode([query])
        
        # Get or compute chunk embeddings (cached)
        model_name = getattr(model, 'model_name', 'original' if 'mpnet' in str(model) else 'fine_tuned')
        cache_file = self.cache_dir / f"{model_name}_embeddings.npy"
        
        if model_name not in self.chunk_embeddings_cache:
            # Try to load from disk cache first
            if cache_file.exists():
                print(f"Loading cached embeddings for {model_name} from disk...")
                chunk_embeddings = np.load(cache_file)
                self.chunk_embeddings_cache[model_name] = chunk_embeddings
                print(f"✅ Loaded {len(chunk_embeddings)} cached embeddings")
            else:
                print(f"Computing embeddings for {len(self.chunks)} chunks with {model_name}...")
                chunk_texts = [chunk.get('text', '') for chunk in self.chunks]
                chunk_embeddings = model.encode(chunk_texts)
                self.chunk_embeddings_cache[model_name] = chunk_embeddings
                # Save to disk cache
                np.save(cache_file, chunk_embeddings)
                print(f"✅ Embeddings computed and cached to {cache_file}")
        else:
            chunk_embeddings = self.chunk_embeddings_cache[model_name]
        
        # Calculate similarities
        similarities = np.dot(query_embedding, chunk_embeddings.T)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = similarities[idx]
            results.append((chunk, score))
        
        return results
    
    def evaluate_relevance(self, results: List[Tuple[Dict, float]], expected_sections: List[str]) -> Dict[str, float]:
        """Evaluate how well the results match expected sections"""
        if not results:
            return {"precision": 0.0, "recall": 0.0, "relevance_score": 0.0}
        
        relevant_found = 0
        total_found = len(results)
        from src.utils.text_utils import extract_header_identifier
        for chunk, score in results:
            header = chunk.get('enriched_headers', '').lower()
            header = [extract_header_identifier(hd) or hd.strip() for hd in header.split(">")]
            header = "> ".join(header)
            
            text = chunk.get('text', '').lower()
            
            # Check if any expected section is mentioned
            for expected in expected_sections:
                if expected.lower() in header or expected.lower() in text:
                    relevant_found += 1
                    break
        
        precision = relevant_found / total_found if total_found > 0 else 0.0
        recall = min(relevant_found / len(expected_sections), 1.0) if expected_sections else 0.0
        relevance_score = (precision + recall) / 2  # Simple F1-like score
        
        return {
            "precision": precision,
            "recall": recall, 
            "relevance_score": relevance_score,
            "relevant_found": relevant_found,
            "total_found": total_found
        }
    
    def display_results(self, question: Dict, original_results: List, fine_tuned_results: List, 
                       original_eval: Dict, fine_tuned_eval: Dict):
        """Display comparison results nicely"""
        print(f"\n{'='*80}")
        print(f"📋 QUESTION: {question['question']}")
        print(f"📂 Category: {question['category']} | 🎯 Difficulty: {question['difficulty']}")
        print(f"🔍 Expected sections: {', '.join(question['expected_sections'])}")
        print(f"{'='*80}")
        
        # Model comparison summary
        print(f"\n📊 EVALUATION SUMMARY:")
        print(f"{'Metric':<20} {'Original':<12} {'Fine-tuned':<12} {'Winner':<10}")
        print(f"{'-'*56}")
        
        # Compare metrics
        metrics = ['precision', 'recall', 'relevance_score']
        winners = {"original": 0, "fine_tuned": 0, "tie": 0}
        
        for metric in metrics:
            orig_val = original_eval[metric]
            ft_val = fine_tuned_eval[metric] if fine_tuned_eval else 0.0
            
            if ft_val > orig_val:
                winner = "Fine-tuned ✅"
                winners["fine_tuned"] += 1
            elif orig_val > ft_val:
                winner = "Original ✅"
                winners["original"] += 1
            else:
                winner = "Tie"
                winners["tie"] += 1
            
            print(f"{metric.title():<20} {orig_val:<12.3f} {ft_val:<12.3f} {winner:<10}")
        
        # Show top chunks from each model
        print(f"\n🔍 TOP CHUNKS FOUND:")
        
        from src.utils.text_utils import extract_header_identifier
        print(f"\n📈 ORIGINAL MODEL RESULTS:")
        for i, (chunk, score) in enumerate(original_results[:3], 1):
            header = chunk.get('enriched_headers', 'No header')
            header = [extract_header_identifier(hd) or hd.strip() for hd in header.split(">")]
            header = ">".join(header)
            text_preview = chunk.get('text', '')[:100].replace('\n', ' ')
            print(f"  {i}. Score: {score:.3f} | {header}")
            print(f"     Preview: {text_preview}...")
        
        if fine_tuned_results:
            print(f"\n🚀 FINE-TUNED MODEL RESULTS:")
            for i, (chunk, score) in enumerate(fine_tuned_results[:3], 1):
                header = chunk.get('enriched_headers', 'No header')
                header = [extract_header_identifier(hd) or hd.strip() for hd in header.split(">")]
                header = ">".join(header)
                text_preview = chunk.get('text', '')[:100].replace('\n', ' ')
                print(f"  {i}. Score: {score:.3f} | {header}")
                print(f"     Preview: {text_preview}...")
        
        return winners
    
    def run_comparison(self):
        """Run complete model comparison"""
        if not self.fine_tuned_model:
            print("❌ Cannot run comparison without fine-tuned model")
            return
        
        questions = self.get_tricky_questions()
        
        print(f"\n🎯 LUXEMBOURG REGULATORY DOCUMENT RETRIEVAL COMPARISON")
        print(f"Comparing {len(questions)} tricky questions")
        print(f"Models: Original vs Fine-tuned")
        
        overall_winners = {"original": 0, "fine_tuned": 0, "tie": 0}
        detailed_results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n\n🔄 Processing Question {i}/{len(questions)}...")
            
            # Get results from both models
            original_results = self.find_relevant_chunks(
                question["question"], self.original_model
            )
            fine_tuned_results = self.find_relevant_chunks(
                question["question"], self.fine_tuned_model
            )
            
            # Evaluate relevance
            original_eval = self.evaluate_relevance(
                original_results, question["expected_sections"]
            )
            fine_tuned_eval = self.evaluate_relevance(
                fine_tuned_results, question["expected_sections"]
            )
            
            # Display results
            question_winners = self.display_results(
                question, original_results, fine_tuned_results,
                original_eval, fine_tuned_eval
            )
            
            # Track overall winners
            for key in overall_winners:
                overall_winners[key] += question_winners[key]
            
            # Store detailed results
            detailed_results.append({
                "question": question,
                "original_eval": original_eval,
                "fine_tuned_eval": fine_tuned_eval,
                "question_winners": question_winners
            })
        
        # Final summary
        self.print_final_summary(overall_winners, detailed_results)
    
    def print_final_summary(self, overall_winners: Dict, detailed_results: List):
        """Print final comparison summary"""
        print(f"\n\n{'='*80}")
        print(f"🏆 FINAL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        total_metrics = sum(overall_winners.values())
        
        print(f"\n📊 OVERALL METRIC WINS:")
        print(f"🤖 Original Model: {overall_winners['original']}/{total_metrics} ({overall_winners['original']/total_metrics*100:.1f}%)")
        print(f"🚀 Fine-tuned Model: {overall_winners['fine_tuned']}/{total_metrics} ({overall_winners['fine_tuned']/total_metrics*100:.1f}%)")
        print(f"🤝 Ties: {overall_winners['tie']}/{total_metrics} ({overall_winners['tie']/total_metrics*100:.1f}%)")
        
        # Determine overall winner
        if overall_winners['fine_tuned'] > overall_winners['original']:
            print(f"\n🎉 WINNER: Fine-tuned Model!")
            print("✅ Fine-tuning improved retrieval for Luxembourg regulatory documents")
        elif overall_winners['original'] > overall_winners['fine_tuned']:
            print(f"\n🤔 WINNER: Original Model")
            print("❌ Fine-tuning may need adjustment or more training data")
        else:
            print(f"\n🤝 RESULT: Tie")
            print("➡️ Both models perform similarly")
        
        # Category breakdown
        category_performance = {}
        for result in detailed_results:
            category = result["question"]["category"]
            if category not in category_performance:
                category_performance[category] = {"original": 0, "fine_tuned": 0, "tie": 0}
            
            # Find best performing model for this question
            orig_score = result["original_eval"]["relevance_score"]
            ft_score = result["fine_tuned_eval"]["relevance_score"]
            
            if ft_score > orig_score:
                category_performance[category]["fine_tuned"] += 1
            elif orig_score > ft_score:
                category_performance[category]["original"] += 1
            else:
                category_performance[category]["tie"] += 1
        
        print(f"\n📈 PERFORMANCE BY CATEGORY:")
        for category, scores in category_performance.items():
            total = sum(scores.values())
            best = max(scores.keys(), key=lambda k: scores[k])
            print(f"  {category}: {best.title()} wins ({scores[best]}/{total})")

def main():
    """Main execution"""
    print("🔍 Luxembourg Regulatory Document Retrieval Comparison")
    print("=" * 60)
    
    comparator = EmbeddingModelComparator()
    comparator.run_comparison()

if __name__ == "__main__":
    main()