import pandas as pd
import numpy as np
import sqlite3
import os
from sklearn.cluster import KMeans
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

DATABASE_PATH = 'ratings.db'

def get_db_connection():
    """Create a database connection"""
    return sqlite3.connect(DATABASE_PATH)

def get_cluster_data():
    """Retrieve cluster data from database"""
    conn = get_db_connection()
    
    # Explicitly cast to integer types
    cluster_ratings = pd.read_sql(
        '''SELECT 
            CAST(cluster_id AS INTEGER) AS user_id, 
            CAST(movie_id AS INTEGER) AS movie_id, 
            mean AS rating 
        FROM cluster_ratings''', 
        conn
    )
    
    user_clusters = pd.read_sql('SELECT * FROM user_clusters', conn)
    conn.close()
    
    return cluster_ratings, user_clusters

def add_gaussian_noise(ratings, noise_scale=0.1):
    """Add Gaussian noise to ratings for privacy"""
    noise = np.random.normal(0, noise_scale, size=len(ratings))
    noisy_ratings = ratings + noise
    return np.clip(noisy_ratings, 1.0, 5.0)

def standardize_column_names(df):
    """Standardize column names to match expected format"""
    column_mapping = {
        'userId': 'user_id',
        'movieId': 'movie_id',
        'rating': 'rating',
        'timestamp': 'timestamp'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Verify required columns exist
    required_columns = ['user_id', 'movie_id', 'rating']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
        
    return df

def process_ratings(ratings, noise_scale=0.1, save_path='saved_models/cluster/'):
    """Process ratings data and store in database and model files"""
    try:
        # Ensure save path exists
        os.makedirs(save_path, exist_ok=True)

        # Create SQLite connection
        conn = get_db_connection()
        
        ratings_df = ratings.copy()

        # Standardize column names
        ratings_df = standardize_column_names(ratings_df)
        
        # Save original ratings
        ratings_df.to_sql('original_ratings', conn, if_exists='replace', index=False)
        
        # Create user-item matrix for clustering
        user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        # Determine number of clusters
        n_users = len(user_item_matrix)
        n_clusters = int(np.sqrt(n_users))
        n_clusters = min(n_clusters, n_users // 30)  # Ensure at least 30 users per cluster
        n_clusters = max(2, min(n_clusters, 100))  # Between 2 and 100 clusters
        
        print(f"Using {n_clusters} clusters for {n_users} users")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans._n_threads = _openmp_effective_n_threads()
        cluster_labels = kmeans.fit_predict(user_item_matrix) + 1  # Make 1-based
        
        # Create cluster mappings and ratings
        user_clusters = pd.DataFrame({
            'user_id': user_item_matrix.index,
            'cluster_id': cluster_labels
        })
        
        # Add clusters to ratings and calculate cluster averages
        ratings_df['cluster_id'] = ratings_df['user_id'].map(
            dict(zip(user_clusters['user_id'], user_clusters['cluster_id']))
        )
        
        cluster_ratings = ratings_df.groupby(['cluster_id', 'movie_id'])['rating'].agg(
            ['mean', 'count']
        ).reset_index()
        
        # Add noise to cluster ratings for privacy
        cluster_ratings['mean'] = add_gaussian_noise(
            cluster_ratings['mean'].values,
            noise_scale=noise_scale
        )
        
        # Save to database
        user_clusters.to_sql('user_clusters', conn, if_exists='replace', index=False)
        cluster_ratings.to_sql('cluster_ratings', conn, if_exists='replace', index=False)
        
        # Save user clusters as JSON in the model save path
        user_clusters_path = os.path.join(save_path, 'user_clusters.json')
        user_clusters.to_json(user_clusters_path, orient='records')
        
        # Save kmeans centers in the model save path
        kmeans_path = os.path.join(save_path, 'kmeans.npy')
        np.save(kmeans_path, kmeans.cluster_centers_)
        
        conn.close()
        
        return {
            'n_clusters': n_clusters,
            'user_clusters_path': user_clusters_path,
            'kmeans_path': kmeans_path
        }
        
    except Exception as e:
        print(f"Error in process_ratings: {str(e)}")
        raise

def clean_data(ratings, min_ratings=5):
    """Remove users with too few ratings"""
    original_size = len(ratings)
    
    # Standardize column names first
    ratings = standardize_column_names(ratings)
    
    user_counts = ratings.groupby('user_id')['movie_id'].count()
    valid_users = user_counts[user_counts >= min_ratings].index
    
    clean_ratings = ratings[ratings['user_id'].isin(valid_users)]
    
    print(f"Removed {original_size - len(clean_ratings)} ratings from users with < {min_ratings} ratings")
    return clean_ratings
