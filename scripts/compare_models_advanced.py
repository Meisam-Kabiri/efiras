#!/usr/bin/env python3
"""
Advanced Model Comparison Script for Luxembourg Regulatory Documents
Compare original vs fine-tuned embeddings on tricky questions
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))


class EmbeddingModelComparator:
    def __init__(self):
        # Load original model
        print("Loading original sentence transformer model...")
        self.original_model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2"
        )

        # Try to load fine-tuned model
        # fine_tuned_path = "models/efiras_financial_embeddings"
        fine_tuned_path = "models/efiras_contrastive_embeddings"
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
        self.cache_dir = Path("data/data_processed/embeddings_cache")
        self.cache_dir.mkdir(exist_ok=True)

    def load_chunks(self) -> List[Dict]:
        """Load processed chunks from Luxembourg document"""
        chunk_file = "data/data_processed/Lux_cssf18_698eng_chunked_blocks.json"
        # chunk_file = "data/data_processed/Lux_cssf18_698eng_processed_blocks.json"

        if not Path(chunk_file).exists():
            print(f"❌ Chunk file not found: {chunk_file}")
            return []

        with open(chunk_file, "r", encoding="utf-8") as f:
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
                "expected_sections": [
                    "Section 4.1.3",
                    "governing body",
                    "multiple mandates",
                ],
            },
            {
                "question": "How should an IFM handle conflicts of interest when managing both UCIs and individual portfolios?",
                "category": "Conflicts of Interest",
                "difficulty": "Very High",
                "expected_sections": [
                    "Sub-section 5.5.7",
                    "conflicts of interest",
                    "portfolio management",
                ],
            },
            {
                "question": "What are the technical infrastructure requirements for business continuity in Luxembourg IFMs?",
                "category": "Technical Requirements",
                "difficulty": "High",
                "expected_sections": [
                    "Section 5.1.2",
                    "business continuity",
                    "technical infrastructure",
                ],
            },
            {
                "question": "Under what circumstances can an IFM delegate the valuation function and what conditions apply?",
                "category": "Delegation",
                "difficulty": "Very High",
                "expected_sections": [
                    "Section 6.6",
                    "valuation function",
                    "delegation",
                ],
            },
            {
                "question": "What specific reporting obligations does an IFM have regarding risk management procedures to the CSSF?",
                "category": "Risk Management",
                "difficulty": "High",
                "expected_sections": [
                    "Sub-section 5.3.1.5",
                    "risk management procedure",
                    "CSSF",
                    "reporting",
                ],
            },
            {
                "question": "How does the principle of proportionality apply to smaller IFMs regarding compliance requirements?",
                "category": "Proportionality",
                "difficulty": "Very High",
                "expected_sections": ["Part V", "proportionality", "compliance"],
            },
            {
                "question": "What are the specific obligations for IFMs regarding EMIR compliance and derivative monitoring?",
                "category": "EMIR Compliance",
                "difficulty": "High",
                "expected_sections": [
                    "Section 5.5.11",
                    "EMIR",
                    "derivatives",
                    "monitoring",
                ],
            },
            {
                "question": "When establishing a branch in another EU country, what notification procedures must Luxembourg IFMs follow?",
                "category": "Cross-border Operations",
                "difficulty": "Very High",
                "expected_sections": [
                    "Sub-chapter 1.1",
                    "branch",
                    "notification",
                    "EU",
                ],
            },
        ]

    def find_relevant_chunks(
        self, query: str, model: SentenceTransformer, top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """Find most relevant chunks using specified model with persistent caching"""
        if not self.chunks:
            return []

        # Get query embedding
        query_embedding = model.encode([query])

        # Get or compute chunk embeddings (cached)
        model_name = getattr(
            model, "model_name", "original" if "mpnet" in str(model) else "fine_tuned"
        )
        cache_file = self.cache_dir / f"{model_name}_embeddings.npy"

        if model_name not in self.chunk_embeddings_cache:
            # Try to load from disk cache first
            if cache_file.exists():
                print(f"Loading cached embeddings for {model_name} from disk...")
                chunk_embeddings = np.load(cache_file)
                self.chunk_embeddings_cache[model_name] = chunk_embeddings
                print(f"✅ Loaded {len(chunk_embeddings)} cached embeddings")
            else:
                print(
                    f"Computing embeddings for {len(self.chunks)} chunks with {model_name}..."
                )
                chunk_texts = [chunk.get("text", "") for chunk in self.chunks]
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

    def evaluate_relevance(
        self, results: List[Tuple[Dict, float]], expected_sections: List[str]
    ) -> Dict[str, float]:
        """Evaluate relevance with binary detection for specific subsections and improved logic"""
        if not results:
            return {
                "binary_section_found": 0.0,
                "exact_section_matches": 0,
                "content_relevance": 0.0,
                "top_chunk_relevance": 0.0,
                "overall_score": 0.0,
            }

        import re

        from src.utils.text_utils import extract_header_identifier

        # Binary detection for specific subsections (e.g., "Section 4.1.3", "Sub-section 5.5.7")
        section_patterns = []
        content_keywords = []

        for expected in expected_sections:
            # Extract section numbers with regex (e.g., "Section 4.1.3", "Sub-section 5.5.7", "Part V")
            section_match = re.search(
                r"(sub-)?section\s+(\d+(?:\.\d+)*)|part\s+([IVX]+)", expected.lower()
            )
            if section_match:
                section_patterns.append(expected.lower())
            else:
                # If not a specific section, treat as content keyword
                content_keywords.append(expected.lower())

        # Evaluate binary section detection (most important metric)
        exact_sections_found = 0
        for chunk, score in results:
            header = chunk.get("enriched_headers", "").lower()
            text = chunk.get("text", "").lower()
            combined_text = f"{header} {text}"

            # Check for exact section matches
            for pattern in section_patterns:
                if pattern in combined_text:
                    exact_sections_found += 1
                    break  # Only count once per chunk

        # Binary section found (1.0 if any expected section found, 0.0 otherwise)
        binary_section_found = 1.0 if exact_sections_found > 0 else 0.0

        # Content relevance for non-section keywords
        content_matches = 0
        if content_keywords:
            for chunk, score in results:
                header = chunk.get("enriched_headers", "").lower()
                text = chunk.get("text", "").lower()
                combined_text = f"{header} {text}"

                keyword_found = False
                for keyword in content_keywords:
                    if keyword in combined_text:
                        keyword_found = True
                        break

                if keyword_found:
                    content_matches += 1
                    break  # Only count once per chunk

        content_relevance = 1.0 if content_matches > 0 else 0.0

        # Top chunk relevance (highest scoring chunk must be relevant)
        top_chunk_relevant = 0.0
        if results:
            top_chunk, top_score = results[0]  # Highest scoring chunk
            header = top_chunk.get("enriched_headers", "").lower()
            text = top_chunk.get("text", "").lower()
            combined_text = f"{header} {text}"

            # Check if top chunk contains any expected content
            for expected in expected_sections:
                if expected.lower() in combined_text:
                    top_chunk_relevant = 1.0
                    break

        # Overall score: weighted combination with binary section detection as most important
        if section_patterns:
            # If we're looking for specific sections, prioritize binary detection
            overall_score = (
                0.7 * binary_section_found
                + 0.2 * content_relevance
                + 0.1 * top_chunk_relevant
            )
        else:
            # If no specific sections, focus on content and top chunk
            overall_score = 0.6 * content_relevance + 0.4 * top_chunk_relevant

        return {
            "binary_section_found": binary_section_found,
            "exact_section_matches": exact_sections_found,
            "content_relevance": content_relevance,
            "top_chunk_relevance": top_chunk_relevant,
            "overall_score": overall_score,
            "total_results": len(results),
            "expected_sections": len(section_patterns),
            "expected_keywords": len(content_keywords),
        }

    def display_results(
        self,
        question: Dict,
        original_results: List,
        fine_tuned_results: List,
        original_eval: Dict,
        fine_tuned_eval: Dict,
    ):
        """Display comparison results with improved metrics"""
        print(f"\n{'='*80}")
        print(f"📋 QUESTION: {question['question']}")
        print(
            f"📂 Category: {question['category']} | 🎯 Difficulty: {question['difficulty']}"
        )
        print(f"🔍 Expected sections: {', '.join(question['expected_sections'])}")
        print(f"{'='*80}")

        # Model comparison summary with new metrics
        print(f"\n📊 EVALUATION SUMMARY:")
        print(f"{'Metric':<25} {'Original':<12} {'Fine-tuned':<12} {'Winner':<15}")
        print(f"{'-'*66}")

        # Compare meaningful metrics only
        key_metrics = [
            "binary_section_found",
            "content_relevance",
            "top_chunk_relevance",
            "overall_score",
        ]
        winners = {"original": 0, "fine_tuned": 0, "tie": 0}

        for metric in key_metrics:
            orig_val = original_eval.get(metric, 0.0)
            ft_val = fine_tuned_eval.get(metric, 0.0) if fine_tuned_eval else 0.0

            if ft_val > orig_val:
                winner = "Fine-tuned ✅"
                winners["fine_tuned"] += 1
            elif orig_val > ft_val:
                winner = "Original ✅"
                winners["original"] += 1
            else:
                winner = "Tie"
                winners["tie"] += 1

            metric_name = metric.replace("_", " ").title()
            print(f"{metric_name:<25} {orig_val:<12.3f} {ft_val:<12.3f} {winner:<15}")

        # Show binary detection results prominently
        orig_sections = original_eval.get("exact_section_matches", 0)
        ft_sections = (
            fine_tuned_eval.get("exact_section_matches", 0) if fine_tuned_eval else 0
        )
        expected_sections = original_eval.get("expected_sections", 0)

        print(f"\n🎯 SECTION DETECTION RESULTS:")
        print(f"Expected specific sections: {expected_sections}")
        print(f"Original model found: {orig_sections}")
        print(f"Fine-tuned model found: {ft_sections}")

        if expected_sections > 0:
            if ft_sections > orig_sections:
                print("✅ Fine-tuned model better at finding specific sections!")
            elif orig_sections > ft_sections:
                print("❌ Original model better at finding specific sections")
            else:
                print("🤝 Both models found same number of sections")

        # Show similarity score comparison for top results
        if original_results and fine_tuned_results:
            orig_top_sim = original_results[0][1]
            ft_top_sim = fine_tuned_results[0][1]
            print(f"\n📊 TOP SIMILARITY SCORES:")
            print(f"Original model top score:    {orig_top_sim:.4f}")
            print(f"Fine-tuned model top score:  {ft_top_sim:.4f}")
            if ft_top_sim > orig_top_sim:
                print(
                    f"✅ Fine-tuned model: +{ft_top_sim - orig_top_sim:.4f} higher similarity"
                )
            elif orig_top_sim > ft_top_sim:
                print(
                    f"❌ Original model: +{orig_top_sim - ft_top_sim:.4f} higher similarity"
                )
            else:
                print("🤝 Same top similarity score")

        # Show top chunks from each model with relevance indicators
        print(f"\n🔍 TOP CHUNKS FOUND:")

        from src.utils.text_utils import extract_header_identifier

        print(f"\n📈 ORIGINAL MODEL RESULTS:")
        self._display_chunk_results(original_results[:3], question["expected_sections"])

        if fine_tuned_results:
            print(f"\n🚀 FINE-TUNED MODEL RESULTS:")
            self._display_chunk_results(
                fine_tuned_results[:3], question["expected_sections"]
            )

        return winners

    def _display_chunk_results(
        self, results: List[Tuple[Dict, float]], expected_sections: List[str]
    ):
        """Helper to display chunk results with relevance indicators"""
        import re

        from src.utils.text_utils import extract_header_identifier

        for i, (chunk, score) in enumerate(results, 1):
            header = chunk.get("enriched_headers", "No header")
            header = [
                extract_header_identifier(hd) or hd.strip() for hd in header.split(">")
            ]
            header = ">".join(header)
            text_preview = chunk.get("text", "")[:100].replace("\n", " ")

            # Check relevance
            combined_text = f"{header} {text_preview}".lower()
            relevance_indicators = []

            for expected in expected_sections:
                if expected.lower() in combined_text:
                    # Check if it's a specific section
                    section_match = re.search(
                        r"(sub-)?section\s+(\d+(?:\.\d+)*)|part\s+([IVX]+)",
                        expected.lower(),
                    )
                    if section_match:
                        relevance_indicators.append(f"🎯 SECTION")
                    else:
                        relevance_indicators.append(f"✅ KEYWORD")
                    break

            relevance_str = (
                " ".join(relevance_indicators)
                if relevance_indicators
                else "❌ NOT_RELEVANT"
            )

            print(f"  {i}. Similarity: {score:.4f} | {relevance_str}")
            print(f"     Header: {header}")
            print(f"     Preview: {text_preview}...")
            print()

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
                question,
                original_results,
                fine_tuned_results,
                original_eval,
                fine_tuned_eval,
            )

            # Track overall winners
            for key in overall_winners:
                overall_winners[key] += question_winners[key]

            # Store detailed results
            detailed_results.append(
                {
                    "question": question,
                    "original_eval": original_eval,
                    "fine_tuned_eval": fine_tuned_eval,
                    "question_winners": question_winners,
                }
            )

        # Final summary
        self.print_final_summary(overall_winners, detailed_results)

    def print_final_summary(self, overall_winners: Dict, detailed_results: List):
        """Print final comparison summary"""
        print(f"\n\n{'='*80}")
        print(f"🏆 FINAL COMPARISON SUMMARY")
        print(f"{'='*80}")

        total_metrics = sum(overall_winners.values())

        print(f"\n📊 OVERALL METRIC WINS:")
        print(
            f"🤖 Original Model: {overall_winners['original']}/{total_metrics} ({overall_winners['original']/total_metrics*100:.1f}%)"
        )
        print(
            f"🚀 Fine-tuned Model: {overall_winners['fine_tuned']}/{total_metrics} ({overall_winners['fine_tuned']/total_metrics*100:.1f}%)"
        )
        print(
            f"🤝 Ties: {overall_winners['tie']}/{total_metrics} ({overall_winners['tie']/total_metrics*100:.1f}%)"
        )

        # Determine overall winner
        if overall_winners["fine_tuned"] > overall_winners["original"]:
            print(f"\n🎉 WINNER: Fine-tuned Model!")
            print(
                "✅ Fine-tuning improved retrieval for Luxembourg regulatory documents"
            )
        elif overall_winners["original"] > overall_winners["fine_tuned"]:
            print(f"\n🤔 WINNER: Original Model")
            print("❌ Fine-tuning may need adjustment or more training data")
        else:
            print(f"\n🤝 RESULT: Tie")
            print("➡️ Both models perform similarly")

        # Category breakdown and section detection performance
        category_performance = {}
        section_detection_performance = {"original": 0, "fine_tuned": 0, "tie": 0}

        for result in detailed_results:
            category = result["question"]["category"]
            if category not in category_performance:
                category_performance[category] = {
                    "original": 0,
                    "fine_tuned": 0,
                    "tie": 0,
                }

            # Use overall_score for category performance
            orig_score = result["original_eval"].get("overall_score", 0.0)
            ft_score = result["fine_tuned_eval"].get("overall_score", 0.0)

            if ft_score > orig_score:
                category_performance[category]["fine_tuned"] += 1
            elif orig_score > ft_score:
                category_performance[category]["original"] += 1
            else:
                category_performance[category]["tie"] += 1

            # Track section detection performance specifically
            orig_section = result["original_eval"].get("binary_section_found", 0.0)
            ft_section = result["fine_tuned_eval"].get("binary_section_found", 0.0)

            if ft_section > orig_section:
                section_detection_performance["fine_tuned"] += 1
            elif orig_section > ft_section:
                section_detection_performance["original"] += 1
            else:
                section_detection_performance["tie"] += 1

        print(f"\n🎯 BINARY SECTION DETECTION PERFORMANCE:")
        total_questions = sum(section_detection_performance.values())
        for model, count in section_detection_performance.items():
            if count > 0:
                percentage = (count / total_questions) * 100
                print(
                    f"  {model.replace('_', '-').title()}: {count}/{total_questions} questions ({percentage:.1f}%)"
                )

        print(f"\n📈 PERFORMANCE BY CATEGORY:")
        for category, scores in category_performance.items():
            total = sum(scores.values())
            best = max(scores.keys(), key=lambda k: scores[k])
            print(
                f"  {category}: {best.replace('_', '-').title()} wins ({scores[best]}/{total})"
            )


def main():
    """Main execution"""
    print("🔍 Luxembourg Regulatory Document Retrieval Comparison")
    print("=" * 60)

    comparator = EmbeddingModelComparator()
    comparator.run_comparison()


if __name__ == "__main__":
    main()
