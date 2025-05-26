"""
Clustering Model for Quality Categorization - POC
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import config

class ClusteringModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=config.N_CLUSTERS, random_state=42)
        self.feature_columns = None
    
    def categorize_quality(self, combined_df):
        """Categorize document quality using clustering"""
        # Prepare features for clustering
        features_df = self.prepare_features(combined_df)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features_df)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(features_scaled)
        
        # Map clusters to quality categories
        quality_categories = self.map_clusters_to_quality(features_df, cluster_labels)
        
        # Add results to original dataframe
        result_df = combined_df.copy()
        result_df['cluster'] = cluster_labels
        result_df['quality_category'] = quality_categories
        
        # Calculate silhouette score
        if len(set(cluster_labels)) > 1:
            silhouette = silhouette_score(features_scaled, cluster_labels)
            print(f"📊 Clustering quality (silhouette score): {silhouette:.3f}")
        
        return result_df
    
    def prepare_features(self, df):
        """Prepare features for clustering"""
        # Select numerical columns for clustering
        feature_columns = [
            'rouge_1_f', 'rouge_2_f', 'rouge_l_f', 'bleu_score',
            'semantic_similarity', 'compression_ratio',
            'relevance_score', 'coherence_score', 'fluency_score',
            'factual_accuracy_score', 'completeness_score', 'overall_score'
        ]
        
        # Keep only columns that exist in the dataframe
        self.feature_columns = [col for col in feature_columns if col in df.columns]
        
        features_df = df[self.feature_columns].copy()
        
        # Handle missing values
        features_df = features_df.fillna(features_df.median())
        
        return features_df
    
    def map_clusters_to_quality(self, features_df, cluster_labels):
        """Map cluster numbers to quality categories based on scores"""
        # Calculate mean overall score per cluster
        cluster_scores = {}
        for cluster_id in set(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = features_df[cluster_mask]
            
            # Use overall_score if available, otherwise use average of all scores
            if 'overall_score' in cluster_data.columns:
                avg_score = cluster_data['overall_score'].mean()
            else:
                # Average all available score columns
                score_cols = [col for col in cluster_data.columns if 'score' in col]
                if score_cols:
                    avg_score = cluster_data[score_cols].mean().mean()
                else:
                    avg_score = cluster_data.mean().mean()
            
            cluster_scores[cluster_id] = avg_score
        
        # Sort clusters by score and assign quality labels
        sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1])
        
        cluster_to_quality = {}
        for i, (cluster_id, score) in enumerate(sorted_clusters):
            if i < len(config.CLUSTER_LABELS):
                cluster_to_quality[cluster_id] = config.CLUSTER_LABELS[i]
            else:
                cluster_to_quality[cluster_id] = config.CLUSTER_LABELS[-1]  # Default to highest
        
        # Map each document to its quality category
        quality_categories = [cluster_to_quality[cluster] for cluster in cluster_labels]
        
        # Print cluster mapping for transparency
        print("\n📊 Cluster to Quality Mapping:")
        for cluster_id, quality in cluster_to_quality.items():
            score = cluster_scores[cluster_id]
            count = sum(1 for label in cluster_labels if label == cluster_id)
            print(f"   Cluster {cluster_id} → {quality} (avg score: {score:.2f}, {count} docs)")
        
        return quality_categories