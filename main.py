#!/usr/bin/env python3
"""
LLM Evaluation Service - Main Entry Point
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict

# Import required modules
from processor import DocumentProcessor
from metrics_evals import MetricsAndEvals
from models import QualityModel
from vector_index import VectorIndex
import config


def main():
    parser = argparse.ArgumentParser(description='LLM Evaluation Service')
    parser.add_argument('--mode', choices=['train', 'inference', 'vector-query', 'vector-query_test'], required=True)
    parser.add_argument('--source', help='PDF source (URL/directory) for training')
    parser.add_argument('--document', help='Single PDF for inference')
    parser.add_argument('--model', default='quality_model.pkl', help='Model file path')
    parser.add_argument('--index-dir', help='Directory with PDFs for vector index')
    parser.add_argument('--query', help='User query for vector search')
    parser.add_argument('--vector-db-dir', default='vector_db', help='Directory to save/load vector index')

    args = parser.parse_args()

    if args.mode == 'train':
        train_workflow(args.source, args.model)
    elif args.mode == 'inference':
        run_inference(args.document, args.model)
    elif args.mode == 'vector-query':
        run_vector_query(args.index_dir, args.query, args.model)
    elif args.mode == 'vector-query_test':
        run_vector_query_test(args.vector_db_dir, args.index_dir, args.query, args.model)


def train_workflow(source: str, model_path: str):
    """Steps 1-4: Training workflow"""
    print("🚀 Starting training workflow...")

    # Step 1: Download PDFs
    print("\n📥 Step 1: Downloading PDFs...")
    processor = DocumentProcessor()
    pdf_paths = processor.download_pdfs(source)
    print(f"✅ Downloaded {len(pdf_paths)} PDFs")

    # Step 2: Generate summaries
    print("\n📝 Step 2: Generating summaries...")
    documents = processor.process_documents(pdf_paths)
    print(f"✅ Generated summaries for {len(documents)} documents")

    # Step 3: Compute metrics
    print("\n📊 Step 3: Computing metrics...")
    metrics_evals = MetricsAndEvals()
    metrics_df = metrics_evals.compute_all_metrics(documents)
    print("✅ Computed evaluation metrics and LLM judge scores")

    # Step 4: Train model
    print("\n🤖 Step 4: Training model...")
    model = QualityModel()
    model.train(metrics_df)
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")

    # Print summary
    print("\n📊 Training Summary:")
    print(f"Documents processed: {len(documents)}")
    print(f"Model performance: {model.get_performance_summary()}")


def run_inference(pdf_path: str, model_path: str):
    """Step 5: Single document inference"""
    print(f"\n🔍 Step 5: Running inference on {pdf_path}")

    # Load model
    model = QualityModel()
    model.load(model_path)

    # Process document
    processor = DocumentProcessor()
    doc = processor.process_single_document(pdf_path)

    # Compute metrics
    metrics_evals = MetricsAndEvals()
    metrics = metrics_evals.evaluate_single_document(doc)

    # Predict quality
    quality_score, confidence = model.predict(metrics)

    # Print results
    print("\n📊 Inference Results:")
    print(f"Document: {doc['name']}")
    print(f"Quality Score: {quality_score:.2f}/10")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nSummaries:")
    print(f"  High-level: {doc['summary_high']}")
    print(f"  Medium-level: {doc['summary_medium'][:200]}...")
    print(f"\nKey Metrics:")
    # Mathematical metrics
    print("  Mathematical:")
    for k in ['avg_rouge', 'avg_semantic', f'{config.JUDGE_SUMMARY_LEVEL}_rougeL']:
        if k in metrics:
            print(f"    {k}: {metrics[k]:.3f}")
    # LLM Judge scores
    print("  LLM Judge:")
    for criterion in config.JUDGE_CRITERIA.keys():
        score_key = f"{criterion}_score"
        if score_key in metrics:
            print(f"    {criterion}: {metrics[score_key]:.1f}/10")
    print(f"    Overall: {metrics.get('overall_score', 0):.1f}/10")


def run_vector_query(index_dir: str, query: str, model_path: str):
    """Step 6: Vector-based query evaluation"""
    print(f"\n🔍 Step 6: Vector search and evaluation")

    # Create/load vector index
    vector_index = VectorIndex()
    index_file = config.VECTOR_INDEX_FILE

    if index_dir:
        # Build new index from directory
        print(f"Building vector index from {index_dir}...")
        vector_index.build_index(index_dir)
        vector_index.save_index(index_file)
        print(f"✅ Saved index to {index_file}")
    else:
        # Try to load existing index
        if os.path.exists(index_file):
            print(f"Loading existing index from {index_file}...")
            vector_index.load_index(index_file)
        else:
            print("❌ No index found. Specify --index-dir to build one.")
            return

    # Get LLM response (simulate or call actual LLM)
    print(f"\nQuery: {query}")
    llm_response = get_llm_response(query)  # This would call your LLM
    print(f"LLM Response: {llm_response[:200]}...")

    # Retrieve nearest references
    references = vector_index.search(query, k=config.VECTOR_SEARCH_K)
    print(f"\nFound {len(references)} relevant references")

    # Load quality model
    model = QualityModel()
    model.load(model_path)

    # Evaluate response quality
    quality_score = model.evaluate_response(llm_response, references)

    print(f"\n📊 Evaluation Results:")
    print(f"Overall Quality Score: {quality_score:.2f}/10")
    print(f"Reference Documents Used:")
    for i, ref in enumerate(references):
        print(f"  {i + 1}. {ref['title']} (similarity: {ref['score']:.3f})")


def run_vector_query_test(vector_db_dir: str, index_dir: str, query: str, model_path: str):
    """Vector query with persistent index storage"""
    print(f"\n🔍 Vector search with persistent storage")

    # Create vector DB directory if needed
    os.makedirs(vector_db_dir, exist_ok=True)
    index_file = os.path.join(vector_db_dir, "vector_index.pkl")

    # Load or build index
    vector_index = VectorIndex()

    if os.path.exists(index_file):
        print(f"📂 Loading existing index from {index_file}")
        try:
            vector_index.load_index(index_file)
            print(f"✅ Loaded index with {len(vector_index.documents)} documents")
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            if index_dir:
                print(f"🔨 Rebuilding index from {index_dir}")
                vector_index.build_index(index_dir)
                vector_index.save_index(index_file)
            else:
                print("❌ No --index-dir specified to rebuild")
                return
    elif index_dir:
        print(f"🔨 Building new index from {index_dir}")
        vector_index.build_index(index_dir)
        vector_index.save_index(index_file)
        print(f"💾 Saved index to {index_file}")
    else:
        print("❌ No index found and no --index-dir specified")
        return

    # Run query
    print(f"\nQuery: {query}")
    llm_response = get_llm_response(query)
    print(f"LLM Response: {llm_response[:200]}...")

    # Search
    references = vector_index.search(query, k=config.VECTOR_SEARCH_K)
    print(f"\nFound {len(references)} relevant references")

    # Evaluate
    model = QualityModel()
    model.load(model_path)
    quality_score = model.evaluate_response(llm_response, references)

    print(f"\n📊 Results:")
    print(f"Quality Score: {quality_score:.2f}/10")
    for i, ref in enumerate(references):
        print(f"{i + 1}. {ref['title']} (similarity: {ref['score']:.3f})")


def get_llm_response(query: str) -> str:
    """Placeholder for LLM response generation"""
    # In real implementation, call your LLM here
    return f"This is a simulated response to: {query}"


if __name__ == "__main__":
    main()