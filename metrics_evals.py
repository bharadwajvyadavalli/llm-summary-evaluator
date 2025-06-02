"""
Combined Metrics and LLM Judge Evaluation
"""

import numpy as np
import pandas as pd
import re
import time
from typing import List, Dict
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import config

class MetricsAndEvals:
    def __init__(self):
        # Metrics components
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        try:
            self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        except:
            self.embedder = None

        # LLM Judge component
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.judge_criteria = list(config.JUDGE_CRITERIA.keys())

    def compute_all_metrics(self, documents: List[Dict]) -> pd.DataFrame:
        """Compute both metrics and LLM evaluations for all documents"""
        all_results = []

        for i, doc in enumerate(documents):
            print(f"Processing {i+1}/{len(documents)}: {doc['name']}")

            # Compute mathematical metrics
            metrics = self.compute_single_metrics(doc)

            # Compute LLM judge evaluations
            evals = self.evaluate_with_judge(doc)

            # Combine results
            combined = {**metrics, **evals}
            combined['document'] = doc['name']

            all_results.append(combined)

        return pd.DataFrame(all_results)

    def compute_single_metrics(self, doc: Dict) -> Dict:
        """Compute mathematical metrics for single document"""
        metrics = {}

        # Compute for each summary level
        for level in ['high', 'medium', 'low']:
            summary = doc[f'summary_{level}']
            reference = doc['abstract']

            # ROUGE scores
            rouge = self.rouge.score(reference, summary)
            metrics[f'{level}_rouge1'] = rouge['rouge1'].fmeasure
            metrics[f'{level}_rouge2'] = rouge['rouge2'].fmeasure
            metrics[f'{level}_rougeL'] = rouge['rougeL'].fmeasure

            # Semantic similarity
            if self.embedder:
                ref_emb = self.embedder.encode([reference])
                sum_emb = self.embedder.encode([summary])
                sim = cosine_similarity(ref_emb, sum_emb)[0][0]
                metrics[f'{level}_semantic_sim'] = sim

            # Length ratio
            metrics[f'{level}_compression'] = len(summary.split()) / len(reference.split())

        # Average scores
        metrics['avg_rouge'] = np.mean([metrics[f'{l}_rougeL'] for l in ['high', 'medium', 'low']])
        metrics['avg_semantic'] = np.mean([metrics[f'{l}_semantic_sim'] for l in ['high', 'medium', 'low']
                                          if f'{l}_semantic_sim' in metrics])

        return metrics

    def evaluate_with_judge(self, doc: Dict) -> Dict:
        """Evaluate document using LLM judge"""
        # Use configured summary level for evaluation
        summary = doc[f'summary_{config.JUDGE_SUMMARY_LEVEL}']
        reference = doc['abstract']

        evals = {}

        # Evaluate each criterion
        for criterion in self.judge_criteria:
            score = self._evaluate_criterion(summary, reference, criterion)
            evals[f'{criterion}_score'] = score
            time.sleep(0.2)  # Rate limiting

        # Calculate overall weighted score
        evals['overall_score'] = self._calculate_overall_score(evals)

        return evals

    def _evaluate_criterion(self, summary: str, reference: str, criterion: str) -> float:
        """Evaluate a single criterion using LLM"""
        criterion_config = config.JUDGE_CRITERIA[criterion]

        prompt = f"""
        {criterion_config['prompt']}
        
        Abstract: {reference}
        
        Summary: {summary}
        
        Provide only a numerical score from 1-10.
        Score: """

        try:
            response = self.client.chat.completions.create(
                model=config.JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )

            response_text = response.choices[0].message.content
            return self._parse_score(response_text)

        except Exception as e:
            print(f"    Warning: Error evaluating {criterion}: {e}")
            return 5.0  # Default score

    def _parse_score(self, response_text: str) -> float:
        """Parse score from LLM response"""
        # Look for numbers in the response
        numbers = re.findall(r'\b([1-9]|10)\b', response_text)

        if numbers:
            score = float(numbers[0])
            return max(1, min(10, score))  # Clamp to 1-10

        return 5.0  # Default if no score found

    def _calculate_overall_score(self, evals: Dict) -> float:
        """Calculate weighted overall score"""
        total_score = 0
        total_weight = 0

        for criterion, config_item in config.JUDGE_CRITERIA.items():
            score = evals.get(f'{criterion}_score')
            if score is not None:
                weight = config_item['weight']
                total_score += score * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0

    def evaluate_single_document(self, doc: Dict) -> Dict:
        """Evaluate a single document (for inference)"""
        # Get mathematical metrics
        metrics = self.compute_single_metrics(doc)

        # Get LLM evaluations
        evals = self.evaluate_with_judge(doc)

        # Combine and return
        return {**metrics, **evals}

    def get_feature_columns(self) -> List[str]:
        """Get all feature column names for model training"""
        # Mathematical metrics
        metric_cols = []
        for level in ['high', 'medium', 'low']:
            for metric in ['rouge1', 'rouge2', 'rougeL', 'semantic_sim', 'compression']:
                metric_cols.append(f'{level}_{metric}')

        # Average metrics
        metric_cols.extend(['avg_rouge', 'avg_semantic'])

        # Judge evaluation scores
        eval_cols = [f'{criterion}_score' for criterion in self.judge_criteria]
        eval_cols.append('overall_score')

        return metric_cols + eval_cols