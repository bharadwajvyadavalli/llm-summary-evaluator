"""
Vector Index for Document Retrieval
"""

import os
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import config

class VectorIndex:
    def __init__(self):
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.index = None
        self.documents = []

    def build_index(self, pdf_dir: str):
        """Build vector index from new documents (not training data)"""
        print(f"Building vector index from {pdf_dir}...")

        # Import here to avoid circular imports
        from processor import DocumentProcessor

        # Process PDFs
        processor = DocumentProcessor()
        pdf_paths = list(os.scandir(pdf_dir))
        pdf_paths = [p.path for p in pdf_paths if p.path.endswith('.pdf')]

        # Extract text and create embeddings
        embeddings = []
        for path in pdf_paths:
            text = processor._extract_text(path)
            abstract = processor._extract_abstract(text) or text[:1000]

            # Store document info
            self.documents.append({
                'path': path,
                'title': os.path.basename(path),
                'text': text[:2000],  # Store first 2000 chars
                'abstract': abstract
            })

            # Create embedding
            embedding = self.embedder.encode(abstract)
            embeddings.append(embedding)

        # Build FAISS index
        embeddings = np.array(embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        print(f"✅ Indexed {len(self.documents)} documents")

    def search(self, query: str, k: int = None) -> List[Dict]:
        """Search for nearest documents to query"""
        if self.index is None or self.index.ntotal == 0:
            print("⚠️  Index is empty. Build index first with --index-dir option.")
            return []

        # Use default from config if not specified
        k = k or config.VECTOR_SEARCH_K

        # Limit k to available documents
        k = min(k, len(self.documents))

        # Encode query
        query_embedding = self.embedder.encode([query]).astype('float32')

        # Search
        distances, indices = self.index.search(query_embedding, k)

        # Return results with scores
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = 1 / (1 + dist)  # Convert distance to similarity
                doc['rank'] = i + 1
                results.append(doc)

        return results

    def save_index(self, filepath: str):
        """Save index to file"""
        data = {
            'documents': self.documents,
            'index': faiss.serialize_index(self.index) if self.index else None
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_index(self, filepath: str):
        """Load index from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.documents = data['documents']
        if data['index'] is not None:
            self.index = faiss.deserialize_index(data['index'])