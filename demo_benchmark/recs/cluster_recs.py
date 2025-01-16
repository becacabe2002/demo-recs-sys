import numpy as np
import pandas as pd
import json
import os
from sklearn.cluster import KMeans
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from .funk_recs import FunkSVDRecs

class ClusterAsUserRecs:
    def __init__(self, ratings, save_path='saved_models/cluster/'):
        self.save_path = save_path
        self.model_loaded = False
        self.ratings = ratings.copy()
        self.kmeans = None
        self.user_cluster_mapping = {}
        self.svd_recommender = None

    def load_model(self, save_path=None):
        """Load model parameters and cluster mappings"""
        if save_path:
            self.save_path = save_path

        # Load user cluster mapping
        with open(os.path.join(self.save_path, 'user_clusters.json'), 'r') as f:
            self.user_cluster_mapping = json.load(f)

        # Load kmeans centers
        centers = np.load(os.path.join(self.save_path, 'kmeans.npy'))
        self.kmeans = KMeans(n_clusters=len(centers), random_state=42)
        self.kmeans._n_threads = _openmp_effective_n_threads()
        self.kmeans.cluster_centers_ = centers

        # Initialize and load SVD recommender
        self.svd_recommender = FunkSVDRecs(
            self.ratings, 
            save_path=os.path.join(self.save_path, 'funk_svd')
        )
        self.svd_recommender.load_model()
        
        self.model_loaded = True

    def get_user_cluster(self, user_id):
        """Get cluster for a user or predict cluster for new user"""
        user_id = int(user_id)
        for entry in self.user_cluster_mapping:
            if entry['user_id'] == user_id:
                print(f"Found user_id {user_id} in {entry['cluster_id']}.")
                return str(entry['cluster_id'])
        
        print(f"Cannot found user_id = {user_id} in cluster map.\n")
        # For new user, predict cluster based on ratings
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        if len(user_ratings) == 0:
            print(f"User id {user_id} has no ratings, so assigned to cluster 1.\n")
            return "1"  # Default to first cluster if no ratings

        # Create user feature vector
        user_vector = pd.pivot_table(
            user_ratings,
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )

        # Predict cluster
        cluster = self.kmeans.predict(user_vector)[0] + 1  # Make 1-based
        print(f"User id {user_id} is predict to belong to cluster {cluster}\n")
        return str(cluster)

    def predict_score(self, user_id, item_id):
        """Predict rating using cluster's predictions"""
        if not self.model_loaded:
            self.load_model()

        # Get user's cluster and use it for prediction
        cluster_id = self.get_user_cluster(user_id)
        return self.svd_recommender.predict_score(cluster_id, item_id)

    def recommend_items(self, user_id, num=6):
        """Get recommendations based on cluster preferences"""
        if not self.model_loaded:
            self.load_model()

        # Get user's rated items
        active_user_items = self.ratings[
            self.ratings['user_id'] == user_id
        ].sort_values(by='rating', ascending=False).iloc[:100]

        return self.recommend_items_by_ratings(user_id, active_user_items, num)

    def recommend_items_by_ratings(self, user_id, active_user_items, num=6):
        """Get recommendations based on cluster and rating history"""
        if not self.model_loaded:
            self.load_model()

        # Get user's cluster
        cluster_id = self.get_user_cluster(user_id)
        
        # Get items to exclude (already rated by user)
        rated_items = set(active_user_items['movie_id'])

        # Get cluster recommendations
        cluster_recs = self.svd_recommender.recommend_items(
            cluster_id, 
            num=num + len(rated_items)
        )

        # Filter out items user has already rated
        filtered_recs = [
            (item_id, score_dict)
            for item_id, score_dict in cluster_recs
            if item_id not in rated_items
        ]

        return filtered_recs[:num]
