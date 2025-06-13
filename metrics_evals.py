"""Metrics Calculation and LLM Judge Evaluation"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import re
import config

class MetricsEvaluator:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    def evaluate_batch(self, documents):
        """Evaluate all documents"""
        results = []

        for doc in documents:
            metrics = self.evaluate_document(doc)
            result = {
                'document': doc['document'],
                **metrics,
                'quality': None  # Will be filled by model
            }

            # Include intended quality if present (for training data)
            if 'summary_quality' in doc:
                result['intended_quality'] = doc['summary_quality']

            results.append(result)

        return results

    def evaluate_document(self, doc):
        """Evaluate single document"""
        summary = doc['summary']
        reference = doc['abstract']

        # Calculate alignment and coverage scores
        alignment_score = self.calculate_alignment_score(summary, reference)
        coverage_score = self.calculate_coverage_score(summary, reference)

        # LLM judge scores
        judge_scores = self.llm_judge_evaluate(summary, reference)

        # Combine all metrics
        return {
            'alignment_score': alignment_score,
            'coverage_score': coverage_score,
            'semantic_similarity': self.calculate_semantic_similarity(summary, reference),
            'compression_ratio': len(summary.split()) / len(reference.split()) if reference else 0,
            **judge_scores,
            'overall_score': self.calculate_overall_score({
                'alignment_score': alignment_score,
                'coverage_score': coverage_score,
                **judge_scores
            })
        }

    def evaluate_answer(self, question, answer, chunks):
        """Evaluate answer quality for Q&A"""
        # Join chunks for reference
        reference = ' '.join(chunks)

        # Calculate alignment and coverage scores
        alignment_score = self.calculate_alignment_score(answer, reference)
        coverage_score = self.calculate_coverage_score(answer, reference)

        # LLM judge for answer quality
        judge_scores = self.llm_judge_answer(question, answer, chunks)

        return {
            'alignment_score': alignment_score,
            'coverage_score': coverage_score,
            'semantic_similarity': self.calculate_semantic_similarity(answer, reference),
            **judge_scores,
            'overall_score': self.calculate_overall_score({
                'alignment_score': alignment_score,
                'coverage_score': coverage_score,
                **judge_scores
            })
        }

    def calculate_alignment_score(self, text, reference):
        """Calculate alignment score using semantic similarity of sentences"""
        try:
            # Split into sentences
            text_sentences = [s.strip() for s in text.split('.') if s.strip()]
            ref_sentences = [s.strip() for s in reference.split('.') if s.strip()]

            if not text_sentences or not ref_sentences:
                return 0.0

            # Encode sentences
            text_embeddings = self.sentence_model.encode(text_sentences)
            ref_embeddings = self.sentence_model.encode(ref_sentences)

            # Calculate alignment as average max similarity
            alignment_scores = []
            for text_emb in text_embeddings:
                # Find best matching reference sentence
                similarities = cosine_similarity([text_emb], ref_embeddings)[0]
                max_similarity = max(similarities) if len(similarities) > 0 else 0
                alignment_scores.append(max_similarity)

            return float(np.mean(alignment_scores)) if alignment_scores else 0.0

        except Exception as e:
            print(f"Alignment calculation error: {e}")
            return 0.0

    def calculate_coverage_score(self, text, reference):
        """Calculate how well the text covers key concepts from reference"""
        try:
            # Extract key phrases (simple approach - can be enhanced)
            ref_words = set(reference.lower().split())
            text_words = set(text.lower().split())

            # Remove common words
            common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but'}
            ref_keywords = ref_words - common_words
            text_keywords = text_words - common_words

            if not ref_keywords:
                return 1.0

            # Calculate coverage
            covered_keywords = ref_keywords.intersection(text_keywords)
            coverage = len(covered_keywords) / len(ref_keywords)

            # Also consider semantic coverage using embeddings
            text_embedding = self.sentence_model.encode([text])
            ref_embedding = self.sentence_model.encode([reference])
            semantic_coverage = cosine_similarity(text_embedding, ref_embedding)[0][0]

            # Combine keyword and semantic coverage
            return float((coverage + semantic_coverage) / 2)

        except Exception as e:
            print(f"Coverage calculation error: {e}")
            return 0.0

    def calculate_semantic_similarity(self, text, reference):
        """Calculate semantic similarity"""
        try:
            text_emb = self.sentence_model.encode([text])
            ref_emb = self.sentence_model.encode([reference])
            similarity = cosine_similarity(text_emb, ref_emb)[0][0]
            return float(similarity)
        except:
            return 0.0

    def llm_judge_evaluate(self, summary, reference):
        """LLM judge evaluation"""
        scores = {}

        for criterion in config.JUDGE_CRITERIA:
            try:
                prompt = self.get_evaluation_prompt(criterion, summary, reference)

                response = self.client.chat.completions.create(
                    model=config.JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.1
                )

                score = self.parse_score(response.choices[0].message.content)
                scores[f'{criterion}_score'] = score

            except Exception as e:
                print(f"Judge error for {criterion}: {e}")
                scores[f'{criterion}_score'] = 5.0

        return scores

    def llm_judge_answer(self, question, answer, chunks):
        """Judge answer quality"""
        prompt = f"""
        Rate this answer on relevance (1-10), coherence (1-10), and accuracy (1-10).
        
        Question: {question}
        Answer: {answer}
        Context: {' '.join(chunks[:2])}
        
        Format: Relevance: X/10, Coherence: Y/10, Accuracy: Z/10
        """

        try:
            response = self.client.chat.completions.create(
                model=config.JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )

            text = response.choices[0].message.content
            relevance = self.parse_score(text, 'relevance')
            coherence = self.parse_score(text, 'coherence')
            accuracy = self.parse_score(text, 'accuracy')

            return {
                'relevance_score': relevance,
                'coherence_score': coherence,
                'accuracy_score': accuracy
            }
        except:
            return {'relevance_score': 5.0, 'coherence_score': 5.0, 'accuracy_score': 5.0}

    def get_evaluation_prompt(self, criterion, summary, reference):
        """Get criterion-specific prompt"""
        prompts = {
            'relevance': f"Rate relevance of summary to abstract (1-10):\nAbstract: {reference}\nSummary: {summary}\nScore:",
            'coherence': f"Rate coherence and flow of summary (1-10):\nSummary: {summary}\nScore:",
            'accuracy': f"Rate factual accuracy of summary compared to abstract (1-10):\nAbstract: {reference}\nSummary: {summary}\nScore:"
        }
        return prompts.get(criterion, f"Rate {criterion} (1-10):\nSummary: {summary}\nScore:")

    def parse_score(self, text, keyword=None):
        """Parse score from text"""
        pattern = f'{keyword}.*?(\\d+)' if keyword else r'(\d+)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return max(1, min(10, score))
        return 5.0

    def calculate_overall_score(self, metrics):
        """Calculate weighted overall score"""
        # Weight the scores (alignment and coverage are 0-1, others are 1-10)
        weights = {
            'alignment_score': 2.0,  # Convert to 10-scale and weight
            'coverage_score': 2.0,   # Convert to 10-scale and weight
            'relevance_score': 1.5,
            'coherence_score': 1.0,
            'accuracy_score': 1.5
        }

        total_score = 0
        total_weight = 0

        for metric, weight in weights.items():
            if metric in metrics:
                value = metrics[metric]
                # Convert alignment and coverage from 0-1 to 0-10 scale
                if metric in ['alignment_score', 'coverage_score']:
                    value = value * 10
                total_score += value * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 5.0