import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

# Configuration
MIN_RESULTS_TO_SHOW = 5  # Minimum number of recommendations to display

@st.cache_data
def load_data():
    """Load and preprocess data"""
    try:
        encoded_df = pd.read_parquet('encoded_data.parquet')
        original_df = pd.read_csv('cleaned_data.csv')
        return original_df, encoded_df
    except Exception as e:
        st.error(f"Error loading data files: {str(e)}")
        st.error("Please make sure 'encoded_data.parquet' and 'cleaned_data.csv' exist")
        return None, None

@st.cache_resource
def load_models():
    """Load precomputed models"""
    try:
        models = {}
        
        # Load scaler
        models['scaler'] = joblib.load('models/scaler.joblib')
        
        # Load clusters
        models['clusters'] = np.load('models/clusters.npy')
        
        # Load NearestNeighbors
        models['nn_model'] = joblib.load('models/nearest_neighbors.joblib')
        
        # Load feature columns
        with open('models/feature_cols.txt', 'r') as f:
            models['feature_cols'] = f.read().splitlines()
        
        return models
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.error("Please run 'python prepare_models.py' first to generate model files")
        return None

def apply_filters(df, city, cuisines, min_rating, min_price, max_price):
    """Apply all user filters to the dataframe"""
    filtered = df[
        (df['city'] == city) &
        (df['cuisine'].isin(cuisines)) &
        (df['rating'] >= min_rating) &
        (df['cost'] >= min_price) &
        (df['cost'] <= max_price)
    ]
    return filtered

def get_similarity_based_recommendations(filtered, encoded_df, original_df, nn_model, scaler, feature_cols):
    """Get recommendations using similarity-based approach with proper filtering"""
    if len(filtered) == 0:
        return pd.DataFrame()
    
    # Get average features for filtered restaurants
    avg_features = scaler.transform(encoded_df[feature_cols].loc[filtered.index].mean().values.reshape(1, -1))
    distances, indices = nn_model.kneighbors(avg_features)
    
    # Get recommendations from full dataset
    recommendations = original_df.iloc[indices[0]]
    
    # Apply filters to recommendations (ensuring they match criteria)
    recommendations = recommendations[
        (recommendations['city'].isin(filtered['city'].unique())) &
        (recommendations['cuisine'].isin(filtered['cuisine'].unique())) &
        (recommendations['rating'] >= filtered['rating'].min()) &
        (recommendations['cost'] >= filtered['cost'].min()) &
        (recommendations['cost'] <= filtered['cost'].max())
    ]
    
    return recommendations

def get_cluster_based_recommendations(filtered, original_df, clusters):
    """Get recommendations using cluster-based approach with proper filtering"""
    if len(filtered) == 0:
        return pd.DataFrame()
    
    # Find most common cluster in filtered results
    filtered_clusters = clusters[filtered.index]
    if len(filtered_clusters) == 0:
        return pd.DataFrame()
    
    most_common_cluster = np.bincount(filtered_clusters).argmax()
    
    # Get restaurants from the same cluster that match all filters
    cluster_indices = np.where(clusters == most_common_cluster)[0]
    recommendations = original_df.iloc[cluster_indices]
    
    # Apply the same filters to cluster results
    recommendations = recommendations[
        (recommendations['city'].isin(filtered['city'].unique())) &
        (recommendations['cuisine'].isin(filtered['cuisine'].unique())) &
        (recommendations['rating'] >= filtered['rating'].min()) &
        (recommendations['cost'] >= filtered['cost'].min()) &
        (recommendations['cost'] <= filtered['cost'].max())
    ]
    
    return recommendations.sort_values('rating', ascending=False)

def main():
    st.title("🍽️ Restaurant Recommendation Engine")
    
    # Load data and models
    original_df, encoded_df = load_data()
    if original_df is None or encoded_df is None:
        return
    
    models = load_models()
    if models is None:
        return
    
    # Extract models
    nn_model = models['nn_model']
    clusters = models['clusters']
    feature_cols = models['feature_cols']
    scaler = models['scaler']
    
    # User Input Section
    st.sidebar.header("Filter Preferences")
    
    # City filter
    selected_city = st.sidebar.selectbox(
        "City",
        original_df['city'].unique()
    )
    
    # Cuisine multiselect
    available_cuisines = original_df['cuisine'].unique()
    selected_cuisines = st.sidebar.multiselect(
        "Cuisine Types",
        available_cuisines,
        default=available_cuisines[:3] if len(available_cuisines) > 3 else available_cuisines
    )
    
    # Rating slider
    min_rating = st.sidebar.slider(
        "Minimum Rating",
        min_value=0.0,
        max_value=5.0,
        value=3.5,
        step=0.5
    )
    
    # Price range slider
    min_price, max_price = st.sidebar.slider(
        "Price Range",
        min_value=0,
        max_value=int(original_df['cost'].max()) + 100,
        value=(0, int(original_df['cost'].max())),
        step=100
    )
    
    # Recommendation method
    method = st.sidebar.radio(
        "Recommendation Method",
        ("Similarity-Based", "Cluster-Based")
    )
    
    # Apply filters
    filtered = apply_filters(original_df, selected_city, selected_cuisines, min_rating, min_price, max_price)
    
    if not filtered.empty:
        if st.button("Get Recommendations"):
            st.subheader(f"Top Recommended Restaurants in {selected_city}")
            
            if method == "Similarity-Based":
                recommendations = get_similarity_based_recommendations(
                    filtered, encoded_df, original_df, nn_model, scaler, feature_cols
                )
            else:  # Cluster-Based
                recommendations = get_cluster_based_recommendations(filtered, original_df, clusters)
            
            # Display results if we have enough
            if len(recommendations) >= MIN_RESULTS_TO_SHOW:
                st.dataframe(
                    recommendations[['name', 'city', 'cuisine', 'rating', 'cost']],
                    column_config={
                        "name": "Restaurant",
                        "city": "City",
                        "cuisine": "Cuisine",
                        "rating": st.column_config.NumberColumn(
                            "Rating",
                            format="%.1f ⭐",
                        ),
                        "cost": st.column_config.NumberColumn(
                            "Cost",
                            format="₹%d",
                        )
                    },
                    hide_index=True
                )
            else:
                st.warning(f"Only found {len(recommendations)} matching recommendations. Try broadening your filters.")
    else:
        st.warning("No restaurants match your filters. Please adjust your criteria.")

if __name__ == '__main__':
    main()