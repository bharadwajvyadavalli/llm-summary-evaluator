"""Vector Search for Query Evaluation"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from pathlib import Path
import openai
import config

class VectorQueryEngine:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.chunks = []
        self.chunk_embeddings = None
        self.chunk_metadata = []

    def index_pdfs(self, pdf_directory):
        """Create vector index from PDFs"""
        print("🔍 Building vector index...")

        pdf_files = list(Path(pdf_directory).glob("*.pdf"))

        for pdf_file in pdf_files:
            text = self._extract_pdf_text(pdf_file)
            if text:
                chunks = self._chunk_text(text, pdf_file.name)
                self.chunks.extend(chunks)

        # Generate embeddings
        if self.chunks:
            texts = [c['text'] for c in self.chunks]
            self.chunk_embeddings = self.encoder.encode(texts)
            print(f"✅ Indexed {len(self.chunks)} chunks from {len(pdf_files)} PDFs")

    def answer_question(self, question):
        """Answer question using vector search and LLM"""
        if not self.chunks:
            return "No documents indexed", []

        # Find relevant chunks
        relevant_chunks = self._search_chunks(question, top_k=5)

        # Generate answer
        answer = self._generate_answer(question, relevant_chunks)

        return answer, [c['text'] for c in relevant_chunks]

    def _extract_pdf_text(self, pdf_path):
        """Extract text from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return None

    def _chunk_text(self, text, source_name, chunk_size=500, overlap=50):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)

            chunks.append({
                'text': chunk_text,
                'source': source_name,
                'index': len(self.chunks) + len(chunks)
            })

        return chunks

    def _search_chunks(self, query, top_k=5):
        """Find most relevant chunks for query"""
        # Encode query
        query_embedding = self.encoder.encode([query])

        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]

        # Get top chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]

        relevant_chunks = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk['score'] = float(similarities[idx])
            relevant_chunks.append(chunk)

        return relevant_chunks

    def _generate_answer(self, question, chunks):
        """Generate answer using LLM"""
        # Prepare context
        context = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)])

        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=config.ANSWER_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer based on the context provided."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Answer generation error: {e}")
            return "Unable to generate answer"

    def get_index_stats(self):
        """Get statistics about the vector index"""
        if not self.chunks:
            return {"status": "No documents indexed"}

        sources = {}
        for chunk in self.chunks:
            source = chunk['source']
            sources[source] = sources.get(source, 0) + 1

        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(sources),
            "chunks_per_document": {k: v for k, v in sources.items()},
            "embedding_dimensions": self.chunk_embeddings.shape[1] if self.chunk_embeddings is not None else 0
        }