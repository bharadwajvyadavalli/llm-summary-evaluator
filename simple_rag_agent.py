"""
Simple RAG Agent - Works with any documents and any queries
"""

import openai
import time
from typing import List, Dict
from dataclasses import dataclass
import uuid
import config


@dataclass
class QueryResponse:
    query: str
    response: str
    context_chunks: List[str]
    sources: List[str]
    timestamp: float
    response_id: str
    context_metadata: List[Dict]


class SimpleRAGAgent:
    """Simple RAG agent for any documents and queries"""

    def __init__(self, vector_manager):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.vector_manager = vector_manager
        self.conversation_history = []

    def query(self, question: str, k: int = 5) -> QueryResponse:
        """Process any query against any documents"""

        print(f"\n🔍 Processing query: '{question}'")

        # Retrieve relevant context
        print("   📖 Retrieving relevant context...")
        context_chunks = self.vector_manager.query_documents(question, k=k)

        print(f"   ✅ Found {len(context_chunks)} relevant chunks")
        for i, chunk in enumerate(context_chunks):
            score = chunk['relevance_score']
            source = chunk['metadata']['document_name']
            print(f"      {i + 1}. {source} (relevance: {score:.3f})")

        # Generate response
        print("   🤖 Generating response...")
        response_text = self._generate_response(question, context_chunks)

        # Create response object
        query_response = QueryResponse(
            query=question,
            response=response_text,
            context_chunks=[chunk['content'] for chunk in context_chunks],
            sources=list(set([chunk['metadata']['document_name'] for chunk in context_chunks])),
            timestamp=time.time(),
            response_id=str(uuid.uuid4()),
            context_metadata=[chunk['metadata'] for chunk in context_chunks]
        )

        # Store in history
        self.conversation_history.append(query_response)

        print(f"   ✅ Generated response ({len(response_text)} characters)")
        return query_response

    def _generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate response using retrieved context"""

        # Prepare context from retrieved chunks
        context_text = "\n\n".join([
            f"[From: {chunk['metadata']['document_name']}]\n{chunk['content']}"
            for chunk in context_chunks
        ])

        # Simple, clear prompt
        prompt = f"""Answer the following question based on the provided document context.

CONTEXT FROM DOCUMENTS:
{context_text}

QUESTION: {query}

INSTRUCTIONS:
- Use the provided context to answer the question
- If the context doesn't contain enough information, say so
- Be clear and helpful in your response
- Mention which documents you're referencing when relevant

ANSWER:"""

        try:
            response = self.client.chat.completions.create(
                model=config.JUDGE_MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that answers questions based on provided document context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"