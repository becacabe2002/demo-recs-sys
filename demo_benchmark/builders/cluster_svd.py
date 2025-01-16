import numpy as np
import pandas as pd
import json
import pickle
import os
from datetime import datetime
from sklearn.cluster import KMeans
from .funk_svd import FunkSVD

class ClusterAsUserSVD:
    def __init__(self, save_path='saved_models/cluster/',
                 n_clusters=None, min_users_per_cluster=30,
                 learning_rate=0.002, max_iterations=50,
                 contribution_prob=0.5,  # Probability for random gradient dropout
                 noise_scale=0.1):  # Scale for Gaussian noise in cluster ratings
        self.save_path = save_path
        self.n_clusters = n_clusters
        self.min_users_per_cluster = min_users_per_cluster
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.contribution_prob = contribution_prob
        self.noise_scale = noise_scale

        self.svd_model = None

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def train(self, ratings, k=20):
        """Train the model with privacy-enhanced gradient dropout"""
        print("Training FunkSVD model on cluster ratings...")

        class PrivacyFunkSVD(FunkSVD):
            def __init__(self, contribution_prob=0.5, **kwargs):
                super().__init__(**kwargs)
                self.contribution_prob = contribution_prob

            def calculate_gds(self, ratings, index_random, factor):
                """Add random gradient dropout to calculate_gds"""
                try:
                    # Select batch using randomized indices
                    rating_batch = ratings[index_random]
                    
                    # Robust index conversion
                    user_indices = np.array([self.u_inx[uid] for uid, _, _ in rating_batch])
                    item_indices = np.array([self.i_inx[iid] for _, iid, _ in rating_batch])
                    actual_ratings = np.array([float(r) for _, _, r in rating_batch])

                    batch_size = 1024
                    for start_idx in range(0, len(rating_batch), batch_size):
                        end_idx = min(start_idx + batch_size, len(rating_batch))
                        batch_slice = slice(start_idx, end_idx)

                        # Get batch indices
                        u_idx = user_indices[batch_slice]
                        i_idx = item_indices[batch_slice]
                        ratings_batch = actual_ratings[batch_slice]

                        # Get current biases for batch
                        user_biases = np.array([self.user_bias[u] for u in u_idx])
                        item_biases = np.array([self.item_bias[i] for i in i_idx])

                        # Calculate predictions and errors for batch
                        predictions = self.predict_batch(u_idx, i_idx)
                        errors = ratings_batch - predictions

                        # Apply dropout mask to gradients
                        dropout_mask = (np.random.random(len(u_idx)) < self.contribution_prob)
                        errors = errors * dropout_mask

                        # Update biases vectorized with dropout
                        user_bias_updates = self.bias_learning_rate * (errors - self.bias_reg * user_biases)
                        item_bias_updates = self.bias_learning_rate * (errors - self.bias_reg * item_biases)

                        # Apply bias updates
                        for u, update in zip(u_idx, user_bias_updates):
                            self.user_bias[u] += update
                        for i, update in zip(i_idx, item_bias_updates):
                            self.item_bias[i] += update

                        # Get current factor values for batch
                        user_factors_batch = self.user_factors[u_idx, factor]
                        item_factors_batch = self.item_factors[i_idx, factor]

                        # Calculate updates with dropout already applied to errors
                        user_factor_updates = self.learning_rate * (
                            errors * item_factors_batch - self.bias_reg * user_factors_batch)
                        item_factor_updates = self.learning_rate * (
                            errors * user_factors_batch - self.bias_reg * item_factors_batch)

                        # Apply factor updates
                        np.add.at(
                            self.user_factors[:, factor], u_idx, user_factor_updates
                        )
                        np.add.at(
                            self.item_factors[:, factor], i_idx, item_factor_updates
                        )

                    return self.calculate_rmse(ratings, factor)
                
                except Exception as e:
                    print("\nError in calculate_gds:")
                    print("Error type:", type(e))
                    print("Error message:", str(e))
                    print("Ratings batch sample:", rating_batch[:5])
                    raise

        # Initialize privacy-enhanced SVD model
        self.svd_model = PrivacyFunkSVD(
            save_path=os.path.join(self.save_path, 'funk_svd'),
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
            contribution_prob=self.contribution_prob
        )

        # Train and save the model
        self.svd_model.train(ratings, k)
