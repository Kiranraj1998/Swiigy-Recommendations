import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import joblib

# Configuration
N_CLUSTERS = 50
N_NEIGHBORS = 100

def prepare_models():
    """Prepare and save all models and data needed for the app"""
    # Load data
    encoded_df = pd.read_parquet('encoded_data.parquet')
    original_df = pd.read_csv('cleaned_data.csv')
    
    # Prepare features
    feature_cols = [col for col in encoded_df.columns if col.startswith(('city_', 'cuisine_'))] + ['rating', 'cost']
    features = encoded_df[feature_cols]
    
    # Build and save scaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Build and save KMeans
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, batch_size=1000)
    clusters = kmeans.fit_predict(scaled_features)
    joblib.dump(kmeans, 'models/kmeans.joblib')
    np.save('models/clusters.npy', clusters)
    
    # Build and save NearestNeighbors
    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='ball_tree')
    nn.fit(scaled_features)
    joblib.dump(nn, 'models/nearest_neighbors.joblib')
    
    # Save feature columns
    with open('models/feature_cols.txt', 'w') as f:
        f.write('\n'.join(feature_cols))

if __name__ == '__main__':
    import os
    os.makedirs('models', exist_ok=True)
    prepare_models()