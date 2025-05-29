"""
RAG-specific evaluation metrics
"""

import openai
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict
import config


class RAGEvaluator:
    """Evaluates RAG responses with comprehensive metrics"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    def evaluate_response(self, query_response) -> Dict[str, float]:
        """Comprehensive evaluation of RAG response"""

        print(f"\n📊 Evaluating response for: '{query_response.query[:50]}...'")

        metrics = {}

        # 1. Context Relevance
        print("   🎯 Evaluating context relevance...")
        context_relevance = self._evaluate_context_relevance(
            query_response.query,
            query_response.context_chunks
        )
        metrics['context_relevance'] = context_relevance
        print(f"      Context Relevance: {context_relevance:.3f}")

        # 2. Answer Faithfulness
        print("   🔒 Evaluating answer faithfulness...")
        faithfulness = self._evaluate_faithfulness(
            query_response.response,
            query_response.context_chunks
        )
        metrics['faithfulness'] = faithfulness
        print(f"      Faithfulness: {faithfulness:.3f}")

        # 3. Answer Relevance
        print("   🎪 Evaluating answer relevance...")
        answer_relevance = self._evaluate_answer_relevance(
            query_response.query,
            query_response.response
        )
        metrics['answer_relevance'] = answer_relevance
        print(f"      Answer Relevance: {answer_relevance:.3f}")

        # 4. Completeness
        print("   📋 Evaluating completeness...")
        completeness = self._evaluate_completeness(
            query_response.query,
            query_response.response,
            query_response.context_chunks
        )
        metrics['completeness'] = completeness
        print(f"      Completeness: {completeness:.3f}")

        # 5. Overall RAG Score
        rag_score = (
                context_relevance * config.RAG_EVALUATION_WEIGHTS['context_relevance'] +
                faithfulness * config.RAG_EVALUATION_WEIGHTS['faithfulness'] +
                answer_relevance * config.RAG_EVALUATION_WEIGHTS['answer_relevance'] +
                completeness * config.RAG_EVALUATION_WEIGHTS['completeness']
        )
        metrics['rag_score'] = rag_score

        print(f"   🏆 Overall RAG Score: {rag_score:.3f}")

        return metrics

    def _evaluate_context_relevance(self, query: str, context_chunks: List[str]) -> float:
        """Evaluate relevance of retrieved context to query"""
        if not context_chunks:
            return 0.0

        # Calculate semantic similarity between query and each chunk
        query_embedding = self.sentence_model.encode([query])
        chunk_embeddings = self.sentence_model.encode(context_chunks)

        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        return float(np.mean(similarities))

    def _evaluate_faithfulness(self, response: str, context_chunks: List[str]) -> float:
        """Evaluate how faithful the response is to the context"""
        if not context_chunks:
            return 0.0

        context_text = "\n\n".join(context_chunks)

        prompt = f"""Rate how faithful this answer is to the provided context on a scale of 1-10.
An answer is faithful if it doesn't contradict the context or make claims not supported by the context.

CONTEXT:
{context_text[:2000]}

ANSWER:
{response}

Rate the faithfulness (1-10) where:
1 = Answer contradicts or makes unsupported claims
10 = Answer is completely faithful to the context

Score:"""

        try:
            llm_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )

            score_text = llm_response.choices[0].message.content
            score = self._parse_score(score_text)
            return score / 10.0  # Normalize to 0-1
        except Exception as e:
            print(f"      ⚠️ Error in faithfulness evaluation: {e}")
            return 0.5

    def _evaluate_answer_relevance(self, query: str, response: str) -> float:
        """Evaluate how relevant the answer is to the query"""
        # Semantic similarity between query and response
        embeddings = self.sentence_model.encode([query, response])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def _evaluate_completeness(self, query: str, response: str, context: List[str]) -> float:
        """Evaluate completeness of the answer"""
        context_text = "\n\n".join(context)

        prompt = f"""Rate how completely this answer addresses the question based on available context (1-10).

QUESTION: {query}

AVAILABLE CONTEXT:
{context_text[:2000]}

ANSWER:
{response}

Rate completeness (1-10) where:
1 = Answer barely addresses the question
10 = Answer comprehensively addresses all aspects of the question

Score:"""

        try:
            llm_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )

            score_text = llm_response.choices[0].message.content
            score = self._parse_score(score_text)
            return score / 10.0  # Normalize to 0-1
        except Exception as e:
            print(f"      ⚠️ Error in completeness evaluation: {e}")
            return 0.5

    def _parse_score(self, response_text: str) -> float:
        """Parse numerical score from LLM response"""
        # Look for numbers in the response
        numbers = re.findall(r'\b([1-9]|10)\b', response_text)

        if numbers:
            score = float(numbers[0])
            return max(1, min(10, score))  # Clamp to 1-10

        return 5.0  # Default score