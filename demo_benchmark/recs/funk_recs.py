import numpy as np
import pandas as pd
import json
import pickle
import os

class FunkSVDRecs:
    def __init__(self, ratings, save_path='saved_models/base_funk/'):
        self.save_path = save_path
        self.model_loaded = False
        self.ratings = ratings.copy()
        
        # Calculate global rating statistics
        self.item_counts = (ratings[['movie_id', 'rating']]
                          .groupby('movie_id')
                          .count()
                          .rename(columns={'rating': 'rating_count'})
                          .reset_index())

        self.item_sum = (ratings[['movie_id', 'rating']]
                       .groupby('movie_id')
                       .sum()
                       .rename(columns={'rating': 'rating_sum'})
                       .reset_index())

        self.avg_rating = self.item_sum['rating_sum'].sum() / self.item_counts['rating_count'].sum()

    def load_model(self, save_path=None):
        """Load model parameters from files"""
        if save_path:
            self.save_path = save_path

        with open(os.path.join(self.save_path, 'user_bias.data'), 'rb') as f:
            self.user_bias = pickle.load(f)
        with open(os.path.join(self.save_path, 'item_bias.data'), 'rb') as f:
            self.item_bias = pickle.load(f)
        with open(os.path.join(self.save_path, 'user_factors.json'), 'r') as f:
            self.user_factors = pd.DataFrame(json.load(f)).T
        with open(os.path.join(self.save_path, 'item_factors.json'), 'r') as f:
            self.item_factors = pd.DataFrame(json.load(f)).T
        
        self.model_loaded = True

    def predict_score(self, user_id, item_id):
        """Predict rating for a user-item pair"""
        if not self.model_loaded:
            self.load_model()

        # Check if rating exists in training data
        ratings_subset = self.ratings[
            (self.ratings['user_id'] == user_id) &
            (self.ratings['movie_id'] == item_id)
        ]

        if len(ratings_subset) > 0:
            return ratings_subset.iloc[0]['rating']

        if str(user_id) not in self.user_factors.columns or \
           str(item_id) not in self.item_factors.columns:
            return 0

        return self._calculate_prediction(user_id, item_id)

    def predict_score_by_ratings(self, item_id, user_ratings):
        """Predict rating based on user's existing ratings"""
        if not self.model_loaded:
            self.load_model()

        if not user_ratings or str(item_id) not in self.item_factors.columns:
            return 0

        # Get a known user_id from user_ratings
        u_id = next(iter(user_ratings.keys()))
        ratings_subset = self.ratings[self.ratings['user_id'] == int(u_id)]

        if len(ratings_subset) == 0:
            return 0

        user_id = ratings_subset.iloc[0]['user_id']

        if str(user_id) not in self.user_factors.columns:
            return 0

        return self._calculate_prediction(user_id, item_id)

    def _calculate_prediction(self, user_id, item_id):
        """Calculate rating prediction"""
        user = self.user_factors[str(user_id)]
        item = self.item_factors[str(item_id)]
        
        # Calculate dot product for latent factors
        base_pred = float(user.dot(item))
        
        # Add biases
        prediction = (self.avg_rating + 
                     self.user_bias[int(user_id)] + 
                     self.item_bias[int(item_id)] + 
                     base_pred)
                     
        return max(1.0, min(5.0, prediction))

    def recommend_items(self, user_id, num=6):
        """Get recommendations for a user"""
        if not self.model_loaded:
            self.load_model()

        # Get user's rated items for exclusion
        active_user_items = self.ratings[
            self.ratings['user_id'] == user_id
        ].sort_values(by='rating', ascending=False).iloc[:100]

        return self.recommend_items_by_ratings(user_id, active_user_items, num)

    def recommend_items_by_ratings(self, user_id, active_user_items, num=6):
        """Get recommendations based on user's rating history"""
        if not self.model_loaded:
            self.load_model()

        # Get rated items to exclude
        rated_movies = set(
            active_user_items['movie_id'].astype(int)
        )

        user_id_str = str(user_id)
        if user_id_str not in self.user_factors.columns:
            print(f"Warning: User {user_id} not found in user factors")
            return []

        # Calculate scores for all unrated items efficiently
        user = self.user_factors[user_id_str].values
        unrated_items = [str(i) for i in self.item_factors.columns 
                        if int(i) not in rated_movies]
        
        if not unrated_items:
            return []

        # Vectorized prediction for all unrated items
        items = self.item_factors[unrated_items].values.T
        base_scores = items @ user
        
        # Add biases
        user_bias = self.user_bias[int(user_id)]
        item_biases = [self.item_bias[int(item)] for item in unrated_items]
        final_scores = base_scores + user_bias + item_biases + self.avg_rating
        final_scores = np.clip(final_scores, 1.0, 5.0)

        # Create recommendations list
        scores = list(zip(map(int, unrated_items), final_scores))
        scores.sort(key=lambda x: x[1], reverse=True)

        return [(item_id, {'prediction': float(score)}) 
                for item_id, score in scores[:num]]
