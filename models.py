"""Quality Assessment Model"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

class QualityModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.cluster_mapping = {}
        self.feature_columns = []

    def fit(self, data):
        """Train quality categorization model"""
        df = pd.DataFrame(data)

        # Select features - updated to use new metrics
        self.feature_columns = [col for col in df.columns if any(
            metric in col for metric in ['alignment', 'coverage', 'score', 'similarity', 'ratio']
        )]

        # Prepare features
        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        # Cluster
        labels = self.kmeans.fit_predict(X_scaled)

        # Map clusters to quality levels
        self._map_clusters_to_quality(df, labels)

        # Add quality labels to data
        for i, item in enumerate(data):
            item['quality'] = self.cluster_mapping[labels[i]]

        # Print results
        self._print_clustering_summary(X_scaled, labels, df)

        return self

    def predict(self, data):
        """Predict quality for new data"""
        df = pd.DataFrame(data)

        # Prepare features
        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)

        # Predict clusters
        labels = self.kmeans.predict(X_scaled)

        # Map to quality
        for i, item in enumerate(data):
            item['quality'] = self.cluster_mapping[labels[i]]

        return data

    def predict_single(self, metrics):
        """Predict quality for single item"""
        # Create dataframe with single row
        df = pd.DataFrame([metrics])

        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Prepare and predict
        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        label = self.kmeans.predict(X_scaled)[0]

        return self.cluster_mapping[label]

    def _map_clusters_to_quality(self, df, labels):
        """Map cluster IDs to quality levels"""
        # Calculate mean overall score per cluster
        cluster_scores = {}

        for cluster_id in set(labels):
            mask = labels == cluster_id
            cluster_data = df[mask]

            # Use overall_score if available
            if 'overall_score' in cluster_data.columns:
                avg_score = cluster_data['overall_score'].mean()
            else:
                # Average of all score columns
                score_cols = [col for col in cluster_data.columns if 'score' in col or 'rouge' in col]
                avg_score = cluster_data[score_cols].mean().mean()

            cluster_scores[cluster_id] = avg_score

        # Sort clusters by score
        sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1])

        # Assign quality labels
        quality_labels = ['Low', 'Medium', 'High']
        for i, (cluster_id, _) in enumerate(sorted_clusters):
            self.cluster_mapping[cluster_id] = quality_labels[i]

    def _print_clustering_summary(self, X_scaled, labels, df):
        """Print clustering summary"""
        # Calculate silhouette score
        if len(set(labels)) > 1:
            silhouette = silhouette_score(X_scaled, labels)
            print(f"\n📊 Clustering Quality (Silhouette): {silhouette:.3f}")

        # Print cluster distribution
        print("\n📈 Quality Distribution:")
        for cluster_id, quality in self.cluster_mapping.items():
            count = sum(1 for label in labels if label == cluster_id)
            print(f"   {quality}: {count} documents")

        # If training data includes intended quality, show alignment
        if 'intended_quality' in df.columns:
            print("\n🎯 Cluster Alignment with Intended Quality:")
            for cluster_id, quality in self.cluster_mapping.items():
                cluster_mask = labels == cluster_id
                cluster_df = df[cluster_mask]

                if len(cluster_df) > 0:
                    intended_dist = cluster_df['intended_quality'].value_counts()
                    print(f"\n   {quality} cluster contains:")
                    for intended, count in intended_dist.items():
                        pct = (count / len(cluster_df)) * 100
                        print(f"      - {intended}: {count} ({pct:.1f}%)")

    def get_feature_importance(self):
        """Get feature importance from cluster centers"""
        if not hasattr(self.kmeans, 'cluster_centers_'):
            return {}

        # Calculate variance of each feature across clusters
        centers = self.kmeans.cluster_centers_
        variances = np.var(centers, axis=0)

        # Normalize and create importance dict
        importance = variances / variances.sum()

        return {feat: imp for feat, imp in zip(self.feature_columns, importance)}