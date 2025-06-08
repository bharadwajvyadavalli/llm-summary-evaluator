"""Metrics Calculation and LLM Judge Evaluation"""

import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import re
import config

class MetricsEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
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

        # Mathematical metrics
        rouge_scores = self.calculate_rouge(summary, reference)
        semantic_sim = self.calculate_semantic_similarity(summary, reference)

        # LLM judge scores
        judge_scores = self.llm_judge_evaluate(summary, reference)

        # Combine all metrics
        return {
            **rouge_scores,
            'semantic_similarity': semantic_sim,
            'compression_ratio': len(summary.split()) / len(reference.split()) if reference else 0,
            **judge_scores,
            'overall_score': self.calculate_overall_score({**rouge_scores, **judge_scores})
        }

    def evaluate_answer(self, question, answer, chunks):
        """Evaluate answer quality for Q&A"""
        # Join chunks for reference
        reference = ' '.join(chunks)

        # Calculate metrics
        rouge_scores = self.calculate_rouge(answer, reference)
        semantic_sim = self.calculate_semantic_similarity(answer, reference)

        # LLM judge for answer quality
        judge_scores = self.llm_judge_answer(question, answer, chunks)

        return {
            **rouge_scores,
            'semantic_similarity': semantic_sim,
            **judge_scores,
            'overall_score': self.calculate_overall_score({**rouge_scores, **judge_scores})
        }

    def calculate_rouge(self, text, reference):
        """Calculate ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, text)
            return {
                'rouge_1': scores['rouge1'].fmeasure,
                'rouge_2': scores['rouge2'].fmeasure,
                'rouge_l': scores['rougeL'].fmeasure
            }
        except:
            return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}

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
        Rate this answer on relevance (1-10) and coherence (1-10).
        
        Question: {question}
        Answer: {answer}
        Context: {' '.join(chunks[:2])}
        
        Format: Relevance: X/10, Coherence: Y/10
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

            return {
                'relevance_score': relevance,
                'coherence_score': coherence
            }
        except:
            return {'relevance_score': 5.0, 'coherence_score': 5.0}

    def get_evaluation_prompt(self, criterion, summary, reference):
        """Get criterion-specific prompt"""
        prompts = {
            'relevance': f"Rate relevance of summary to abstract (1-10):\nAbstract: {reference}\nSummary: {summary}\nScore:",
            'coherence': f"Rate coherence and flow of summary (1-10):\nSummary: {summary}\nScore:",
            'completeness': f"Rate completeness of summary vs abstract (1-10):\nAbstract: {reference}\nSummary: {summary}\nScore:"
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
        # Simple average of available scores
        scores = [v for k, v in metrics.items() if 'score' in k or 'rouge' in k]
        return np.mean(scores) * 10 if scores else 5.0