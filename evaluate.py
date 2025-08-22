#!/usr/bin/env python3
"""
Standalone clustering evaluation script - OPTIMIZED FOR LARGE DATASETS
Run: python evaluate_clustering.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import json
import time

def main():
    # Load necessary data
    try:
        print("📂 Loading data...")
        start_time = time.time()
        encoded_df = pd.read_parquet('encoded_data.parquet')
        original_df = pd.read_csv('cleaned_data.csv')
        clusters = np.load('models/clusters.npy')
        kmeans = joblib.load('models/kmeans.joblib')
        load_time = time.time() - start_time
        print(f"✅ Data loaded successfully in {load_time:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Please run data preprocessing and model preparation first")
        return

    # ✅ FIX: Select only numerical features for evaluation
    print("🔍 Selecting numerical features...")
    numerical_cols = encoded_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numerical_cols if col not in ['id']]
    features = encoded_df[feature_cols].values

    print(f"📊 Original features: {features.shape[1]}")
    print(f"📊 Samples: {features.shape[0]}")

    # ✅ CRITICAL: Dimensionality reduction for large feature set
    if features.shape[1] > 100:
        print("📉 Applying PCA for dimensionality reduction...")
        pca_start = time.time()
        pca = PCA(n_components=0.95, random_state=42)
        features = pca.fit_transform(features)
        pca_time = time.time() - pca_start
        print(f"   Reduced to {features.shape[1]} features in {pca_time:.1f}s")
        print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Comprehensive evaluation
    print("\n" + "="*60)
    print("COMPREHENSIVE CLUSTERING EVALUATION REPORT")
    print("="*60)

    # 1. Basic metrics
    print(f"\n📊 BASIC METRICS:")
    unique_clusters = len(np.unique(clusters))
    print(f"   Number of clusters: {unique_clusters}")
    print(f"   Number of restaurants: {len(clusters):,}")
    print(f"   Inertia (WCSS): {kmeans.inertia_:.2f}")
    print(f"   Inertia per sample: {kmeans.inertia_ / len(clusters):.2f}")

    # 2. Cluster size analysis
    print(f"\n📈 CLUSTER SIZE ANALYSIS:")
    cluster_sizes = pd.Series(clusters).value_counts()
    print(f"   Largest cluster: {cluster_sizes.max():,} restaurants")
    print(f"   Smallest cluster: {cluster_sizes.min():,} restaurants")
    print(f"   Average cluster size: {cluster_sizes.mean():.1f}")
    print(f"   Size standard deviation: {cluster_sizes.std():.1f}")

    # Check for empty clusters
    empty_clusters = sum(1 for size in cluster_sizes if size == 0)
    if empty_clusters > 0:
        print(f"   ⚠️  Empty clusters: {empty_clusters}")

    # 3. Validation metrics (optimized for large datasets)
    print(f"\n🎯 VALIDATION METRICS:")
    
    # Use sampling for large datasets
    sample_size = min(5000, len(clusters))  # Sample size
    if len(clusters) > sample_size:
        print(f"   ⚡ Using sampling ({sample_size} samples) for large dataset")
        indices = np.random.choice(len(clusters), sample_size, replace=False)
        sample_features = features[indices]
        sample_clusters = clusters[indices]
    else:
        sample_features = features
        sample_clusters = clusters

    # Calculate metrics with progress indication
    try:
        print("   Calculating Davies-Bouldin Score...")
        db_score = davies_bouldin_score(sample_features, sample_clusters)
        print(f"   ✓ Davies-Bouldin Score: {db_score:.3f}")
        print(f"     → Interpretation: {'Excellent' if db_score < 0.5 else 'Good' if db_score < 1.0 else 'Reasonable' if db_score < 2.0 else 'Poor'}")
    except Exception as e:
        print(f"   ❌ Davies-Bouldin failed: {e}")
        db_score = None

    # For very large datasets, skip expensive metrics but provide alternatives
    if len(clusters) > 10000:
        print("   ⏭️  Silhouette Score: Skipped (too computationally expensive)")
        print("   ⏭️  Calinski-Harabasz: Skipped (too computationally expensive)")
        print("   💡 For large datasets, Davies-Bouldin is the most practical metric")
    else:
        try:
            print("   Calculating Silhouette Score...")
            silhouette = silhouette_score(sample_features, sample_clusters)
            print(f"   ✓ Silhouette Score: {silhouette:.3f}")
            print(f"     → Interpretation: {'Excellent' if silhouette > 0.7 else 'Good' if silhouette > 0.5 else 'Reasonable' if silhouette > 0.25 else 'Weak'}")
        except Exception as e:
            print(f"   ❌ Silhouette Score failed: {e}")
            silhouette = None

        try:
            print("   Calculating Calinski-Harabasz Score...")
            ch_score = calinski_harabasz_score(sample_features, sample_clusters)
            print(f"   ✓ Calinski-Harabasz Score: {ch_score:.1f}")
        except Exception as e:
            print(f"   ❌ Calinski-Harabasz failed: {e}")
            ch_score = None

    # 4. Business context validation
    print(f"\n🍽️ BUSINESS CONTEXT VALIDATION:")
    cluster_df = original_df.copy()
    cluster_df['cluster'] = clusters
    
    # Show statistics for first 5 non-empty clusters
    shown_clusters = 0
    print("   Top 5 clusters by size:")
    for cluster_id in cluster_sizes.head(5).index:
        cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            print(f"\n   Cluster {cluster_id} ({len(cluster_data):,} restaurants):")
            print(f"     Avg rating: {cluster_data['rating'].mean():.2f} ⭐")
            print(f"     Avg cost: ₹{cluster_data['cost'].mean():.2f}")
            
            # Get top 3 cuisines
            top_cuisines = cluster_data['cuisine'].value_counts().head(3)
            if not top_cuisines.empty:
                cuisine_str = ", ".join([f"{cuisine}({count})" for cuisine, count in top_cuisines.items()])
                print(f"     Top cuisines: {cuisine_str}")
            
            shown_clusters += 1

    # 5. Save detailed report
    save_detailed_report(features, clusters, kmeans, cluster_df, 
                        silhouette if 'silhouette' in locals() else None,
                        ch_score if 'ch_score' in locals() else None,
                        db_score if 'db_score' in locals() else None)

def save_detailed_report(features, clusters, kmeans, cluster_df, silhouette, ch_score, db_score):
    """Save comprehensive evaluation report"""
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "dataset_statistics": {
            "n_samples": int(len(clusters)),
            "n_features_original": int(features.shape[1] if hasattr(features, 'shape') else 0),
            "n_clusters": int(len(np.unique(clusters)))
        },
        "clustering_metrics": {
            "silhouette_score": float(silhouette) if silhouette is not None else None,
            "calinski_harabasz_score": float(ch_score) if ch_score is not None else None,
            "davies_bouldin_score": float(db_score) if db_score is not None else None,
            "inertia": float(kmeans.inertia_),
            "inertia_per_sample": float(kmeans.inertia_ / len(clusters)) if len(clusters) > 0 else None
        },
        "cluster_statistics": {
            "cluster_size_distribution": pd.Series(clusters).value_counts().to_dict(),
            "size_statistics": {
                "mean": float(pd.Series(clusters).value_counts().mean()),
                "std": float(pd.Series(clusters).value_counts().std()),
                "min": int(pd.Series(clusters).value_counts().min()),
                "max": int(pd.Series(clusters).value_counts().max()),
                "empty_clusters": int(sum(1 for size in pd.Series(clusters).value_counts() if size == 0))
            }
        },
        "business_metrics": get_business_metrics(cluster_df),
        "evaluation_notes": "For large datasets, some metrics may use sampling or be skipped due to computational constraints"
    }
    
    with open('clustering_evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Detailed report saved to: clustering_evaluation_report.json")

def get_business_metrics(cluster_df):
    """Extract business-relevant metrics"""
    metrics = {}
    cluster_sizes = cluster_df['cluster'].value_counts()
    
    # Only include substantial clusters (more than 100 restaurants)
    for cluster_id in cluster_sizes[cluster_sizes > 100].index:
        cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            metrics[int(cluster_id)] = {
                "size": int(len(cluster_data)),
                "average_rating": float(cluster_data['rating'].mean()),
                "average_cost": float(cluster_data['cost'].mean()),
                "rating_std": float(cluster_data['rating'].std()),
                "cost_std": float(cluster_data['cost'].std()),
                "top_cuisines": cluster_data['cuisine'].value_counts().head(3).to_dict(),
                "top_cities": cluster_data['city'].value_counts().head(3).to_dict(),
                "rating_distribution": {
                    "min": float(cluster_data['rating'].min()),
                    "max": float(cluster_data['rating'].max()),
                    "median": float(cluster_data['rating'].median())
                }
            }
    return metrics

if __name__ == "__main__":
    main()