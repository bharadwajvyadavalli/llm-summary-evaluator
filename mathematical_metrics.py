"""
Mathematical Metrics Calculator for LLM Evaluation POC
"""

import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class MathematicalMetrics:
    def __init__(self):
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            print("Warning: Could not load sentence transformer model")
            self.sentence_model = None
        
        # BLEU smoothing
        self.smoothing = SmoothingFunction().method1
    
    def evaluate_batch(self, documents):
        """Evaluate mathematical metrics for all documents"""
        results = []
        
        for doc in documents:
            summary = doc['summary']
            reference = doc['abstract']
            
            metrics = {
                'document_name': doc['document_name']
            }
            
            # ROUGE scores
            rouge_scores = self.calculate_rouge(summary, reference)
            metrics.update(rouge_scores)
            
            # BLEU score
            bleu_score = self.calculate_bleu(summary, reference)
            metrics['bleu_score'] = bleu_score
            
            # Semantic similarity
            semantic_sim = self.calculate_semantic_similarity(summary, reference)
            metrics['semantic_similarity'] = semantic_sim
            
            # Length metrics
            metrics['compression_ratio'] = len(summary.split()) / len(reference.split()) if reference else 0
            
            results.append(metrics)
        
        return results
    
    def calculate_rouge(self, summary, reference):
        """Calculate ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, summary)
            return {
                'rouge_1_f': scores['rouge1'].fmeasure,
                'rouge_2_f': scores['rouge2'].fmeasure,
                'rouge_l_f': scores['rougeL'].fmeasure
            }
        except:
            return {'rouge_1_f': 0.0, 'rouge_2_f': 0.0, 'rouge_l_f': 0.0}
    
    def calculate_bleu(self, summary, reference):
        """Calculate BLEU score"""
        try:
            # Tokenize
            reference_tokens = nltk.word_tokenize(reference.lower())
            summary_tokens = nltk.word_tokenize(summary.lower())
            
            # Calculate BLEU-4
            bleu = sentence_bleu([reference_tokens], summary_tokens, 
                               weights=(0.25, 0.25, 0.25, 0.25),
                               smoothing_function=self.smoothing)
            return bleu
        except:
            return 0.0
    
    def calculate_semantic_similarity(self, summary, reference):
        """Calculate semantic similarity using embeddings"""
        if not self.sentence_model:
            return 0.0
        
        try:
            # Generate embeddings
            summary_embedding = self.sentence_model.encode([summary])
            reference_embedding = self.sentence_model.encode([reference])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(summary_embedding, reference_embedding)[0][0]
            return float(similarity)
        except:
            return 0.0