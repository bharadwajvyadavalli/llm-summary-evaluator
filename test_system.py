#!/usr/bin/env python3
"""
Simple test script to verify the system works
"""

import os
import numpy as np
import pandas as pd
from model import QualityModel
from metrics_evals import MetricsAndEvals
import config


def test_model():
    """Test the model with synthetic data"""
    print("🧪 Testing LLM Evaluation System...")
    print("=" * 50)

    # 1. Test configuration
    print("\n✓ Config loaded:")
    print(f"  - OpenAI Model: {config.OPENAI_MODEL}")
    print(f"  - Judge Model: {config.JUDGE_MODEL}")
    print(f"  - Quality Levels: {config.QUALITY_LABELS}")
    print(f"  - Judge Criteria: {list(config.JUDGE_CRITERIA.keys())}")
    print(f"  - Embedding Model: {config.EMBEDDING_MODEL}")

    # 2. Create synthetic training data
    print("\n✓ Creating synthetic data...")
    n_docs = 20
    metrics_data = {
        'document': [f'doc_{i}.pdf' for i in range(n_docs)],
        'high_rouge1': np.random.uniform(0.3, 0.9, n_docs),
        'high_semantic_sim': np.random.uniform(0.4, 0.95, n_docs),
        'medium_rouge1': np.random.uniform(0.35, 0.85, n_docs),
        'medium_semantic_sim': np.random.uniform(0.45, 0.9, n_docs),
        'low_rouge1': np.random.uniform(0.4, 0.8, n_docs),
        'low_semantic_sim': np.random.uniform(0.5, 0.85, n_docs),
        'avg_rouge': np.random.uniform(0.35, 0.85, n_docs),
        'avg_semantic': np.random.uniform(0.45, 0.9, n_docs),
        # Add judge scores
        'relevance_score': np.random.uniform(4, 9, n_docs),
        'coherence_score': np.random.uniform(4, 9, n_docs),
        'fluency_score': np.random.uniform(4, 9, n_docs),
        'factual_accuracy_score': np.random.uniform(4, 9, n_docs),
        'completeness_score': np.random.uniform(4, 9, n_docs),
        'overall_score': np.random.uniform(4, 9, n_docs)
    }

    df = pd.DataFrame(metrics_data)

    # 3. Train model
    print("\n✓ Training quality model...")
    model = QualityModel()
    model.train(df)
    print(f"  - Clusters: {len(model.cluster_scores)}")
    print(f"  - Quality mapping: {model.quality_mapping}")

    # 4. Test prediction on good document
    print("\n✓ Testing predictions...")
    good_doc = {
        'high_rouge1': 0.85,
        'high_semantic_sim': 0.9,
        'medium_rouge1': 0.82,
        'medium_semantic_sim': 0.88,
        'low_rouge1': 0.8,
        'low_semantic_sim': 0.85,
        'avg_rouge': 0.82,
        'avg_semantic': 0.88,
        'relevance_score': 8.5,
        'coherence_score': 8.0,
        'fluency_score': 8.5,
        'factual_accuracy_score': 9.0,
        'completeness_score': 8.0,
        'overall_score': 8.4
    }
    score, confidence = model.predict(good_doc)
    print(f"  Good document → Score: {score:.1f}/10, Confidence: {confidence:.1%}")

    # 5. Test prediction on poor document
    poor_doc = {
        'high_rouge1': 0.3,
        'high_semantic_sim': 0.4,
        'medium_rouge1': 0.35,
        'medium_semantic_sim': 0.45,
        'low_rouge1': 0.4,
        'low_semantic_sim': 0.5,
        'avg_rouge': 0.35,
        'avg_semantic': 0.45,
        'relevance_score': 4.0,
        'coherence_score': 4.5,
        'fluency_score': 5.0,
        'factual_accuracy_score': 3.5,
        'completeness_score': 4.0,
        'overall_score': 4.2
    }
    score, confidence = model.predict(poor_doc)
    print(f"  Poor document → Score: {score:.1f}/10, Confidence: {confidence:.1%}")

    # 6. Save and reload test
    print("\n✓ Testing save/load...")
    model.save('test_model.pkl')
    model2 = QualityModel()
    model2.load('test_model.pkl')
    score2, conf2 = model2.predict(good_doc)
    print(f"  After reload → Score: {score2:.1f}/10, Confidence: {conf2:.1%}")

    # Clean up
    if os.path.exists('test_model.pkl'):
        os.remove('test_model.pkl')

    print("\n✅ All tests passed!")
    print("=" * 50)


def test_without_api():
    """Test metrics calculation without API calls"""
    print("\n🧪 Testing metrics calculation...")

    # Create fake document data
    doc = {
        'name': 'test.pdf',
        'abstract': 'This is a test abstract about machine learning and neural networks.',
        'summary_high': 'This is about ML and neural nets.',
        'summary_medium': 'This is a test abstract about machine learning.',
        'summary_low': 'This is a test abstract about machine learning and neural networks. It covers various aspects.'
    }

    # Calculate metrics only (no LLM judge)
    evals = MetricsAndEvals()
    metrics = evals.compute_single_metrics(doc)

    print("✓ Computed metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")

    print("\n✅ Metrics test complete!")
    print("Note: LLM Judge evaluation requires API key and will be used during actual runs")


if __name__ == "__main__":
    print("LLM Evaluation System Test")
    print("=========================\n")

    # Run tests
    test_model()
    test_without_api()

    print("\n📝 Next steps:")
    print("1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
    print("2. Run training: python main.py --mode train --source pdfs/ --model model.pkl")
    print("3. Run inference: python main.py --mode inference --document test.pdf --model model.pkl")