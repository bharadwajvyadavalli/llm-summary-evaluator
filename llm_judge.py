"""
LLM Judge for Subjective Evaluation - POC
"""

import openai
import re
import time
import config

class LLMJudge:
    def __init__(self):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.criteria = list(config.JUDGE_CRITERIA.keys())
    
    def evaluate_batch(self, documents):
        """Evaluate all documents using LLM judge"""
        results = []
        
        for doc in documents:
            print(f"Judging: {doc['document_name']}")
            
            doc_results = {
                'document_name': doc['document_name']
            }
            
            # Evaluate each criterion
            for criterion in self.criteria:
                score = self.evaluate_criterion(doc['summary'], doc['abstract'], criterion)
                doc_results[f'{criterion}_score'] = score
                time.sleep(0.5)  # Rate limiting
            
            # Calculate overall weighted score
            overall_score = self.calculate_overall_score(doc_results)
            doc_results['overall_score'] = overall_score
            
            results.append(doc_results)
        
        return results
    
    def evaluate_criterion(self, summary, reference, criterion):
        """Evaluate a single criterion"""
        prompt = self.get_prompt(criterion, summary, reference)
        
        try:
            response = self.client.chat.completions.create(
                model=config.JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            score = self.parse_score(response_text)
            return score
            
        except Exception as e:
            print(f"Error evaluating {criterion}: {e}")
            return 5.0  # Default score
    
    def get_prompt(self, criterion, summary, reference):
        """Get evaluation prompt for criterion"""
        prompts = {
            "relevance": f"""
            Rate the RELEVANCE of this summary compared to the abstract on a scale of 1-10.
            
            Abstract: {reference}
            Summary: {summary}
            
            Score (1-10): """,
            
            "coherence": f"""
            Rate the COHERENCE and logical flow of this summary on a scale of 1-10.
            
            Summary: {summary}
            
            Score (1-10): """,
            
            "fluency": f"""
            Rate the FLUENCY and language quality of this summary on a scale of 1-10.
            
            Summary: {summary}
            
            Score (1-10): """,
            
            "factual_accuracy": f"""
            Rate the FACTUAL ACCURACY of this summary compared to the abstract on a scale of 1-10.
            
            Abstract: {reference}
            Summary: {summary}
            
            Score (1-10): """,
            
            "completeness": f"""
            Rate the COMPLETENESS of this summary compared to the abstract on a scale of 1-10.
            
            Abstract: {reference}
            Summary: {summary}
            
            Score (1-10): """
        }
        
        return prompts.get(criterion, "Rate this summary on a scale of 1-10.")
    
    def parse_score(self, response_text):
        """Parse score from LLM response"""
        # Look for numbers in the response
        numbers = re.findall(r'\b([1-9]|10)\b', response_text)
        
        if numbers:
            score = float(numbers[0])
            return max(1, min(10, score))  # Clamp to 1-10
        
        return 5.0  # Default if no score found
    
    def calculate_overall_score(self, results):
        """Calculate weighted overall score"""
        total_score = 0
        total_weight = 0
        
        for criterion, weight in config.JUDGE_CRITERIA.items():
            score = results.get(f'{criterion}_score')
            if score is not None:
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0