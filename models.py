"""
Quality Model - Training and prediction
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
import config

class QualityModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=config.N_CLUSTERS, random_state=42)
        self.cluster_scores = {}
        self.feature_cols = []

    def train(self, metrics_df: pd.DataFrame):
        """Train unsupervised model on metrics"""
        # Select feature columns - include both metrics and eval scores
        potential_cols = [
            'rouge', 'semantic', 'compression', 'avg',
            '_score', 'overall_score'  # Include judge scores
        ]
        self.feature_cols = [col for col in metrics_df.columns
                           if any(x in col for x in potential_cols)]

        # Prepare features
        X = metrics_df[self.feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        # Cluster
        labels = self.kmeans.fit_predict(X_scaled)

        # Calculate cluster quality scores
        for i in range(3):
            mask = labels == i
            cluster_data = X[mask]
            self.cluster_scores[i] = cluster_data.mean().mean()

        # Sort clusters by quality
        sorted_clusters = sorted(self.cluster_scores.items(), key=lambda x: x[1])
        self.quality_mapping = {}
        for i, (cluster_id, _) in enumerate(sorted_clusters):
            self.quality_mapping[cluster_id] = config.QUALITY_LABELS[i]

    def predict(self, metrics: Dict) -> Tuple[float, float]:
        """Predict quality score and confidence"""
        # Convert to dataframe row
        df = pd.DataFrame([metrics])
        X = df[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)

        # Predict cluster
        cluster = self.kmeans.predict(X_scaled)[0]

        # Calculate score (0-10 scale)
        base_score = self.cluster_scores[cluster]
        quality_score = min(10, max(0, base_score * 10))

        # Calculate confidence based on distance to cluster center
        distances = self.kmeans.transform(X_scaled)[0]
        confidence = 1 / (1 + distances[cluster])

        return quality_score, confidence

    def evaluate_response(self, response: str, references: List[Dict]) -> float:
        """Evaluate LLM response against references"""
        if not references:
            return 5.0  # Default middle score

        # Simple evaluation based on reference similarity
        # In practice, you'd compute actual metrics here
        scores = [ref['score'] for ref in references]
        base_score = np.mean(scores) * 10

        # Adjust based on response length and quality heuristics
        response_words = len(response.split())
        if 50 <= response_words <= 500:
            base_score += 0.5

        return min(10, max(0, base_score))

    def save(self, filepath: str):
        """Save model to pickle file"""
        model_data = {
            'scaler': self.scaler,
            'kmeans': self.kmeans,
            'cluster_scores': self.cluster_scores,
            'quality_mapping': self.quality_mapping,
            'feature_cols': self.feature_cols
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, filepath: str):
        """Load model from pickle file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.scaler = model_data['scaler']
        self.kmeans = model_data['kmeans']
        self.cluster_scores = model_data['cluster_scores']
        self.quality_mapping = model_data.get('quality_mapping', {})
        self.feature_cols = model_data['feature_cols']

    def get_performance_summary(self) -> str:
        """Get model performance summary"""
        return f"Clusters: {len(self.cluster_scores)}, " \
               f"Quality mapping: {self.quality_mapping}"