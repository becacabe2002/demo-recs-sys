import numpy as np
import pandas as pd
import json
import pickle
import os
import random
from datetime import datetime
from collections import defaultdict

class FunkSVD:
    def __init__(self, save_path='saved_models/base_funk/',
                 learning_rate=0.002,
                 bias_learning_rate=0.005,
                 bias_reg=0.001,
                 max_iterations=50):
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.bias_learning_rate = bias_learning_rate
        self.bias_reg = bias_reg
        self.max_iterations = max_iterations

        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.avg_rating = 0

        self.u_inx = None
        self.i_inx = None
        self.user_ids = None
        self.movie_ids = None

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def initialize_factors(self, ratings, k=25):
        """Initialize model parameters"""
        # Ensure consistent type conversion
        ratings['user_id'] = ratings['user_id'].astype(np.int64)
        ratings['movie_id'] = ratings['movie_id'].astype(np.int64)

        # Identify unique users and movies
        self.user_ids = set(ratings['user_id'].unique())
        self.movie_ids = set(ratings['movie_id'].unique())
        
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

        # Initialize bias dictionaries
        self.user_bias = defaultdict(lambda: 0)
        self.item_bias = defaultdict(lambda: 0)

        # Create index mappings with sorted unique IDs
        sorted_user_ids = sorted(self.user_ids)
        sorted_movie_ids = sorted(self.movie_ids)
        self.u_inx = {np.int64(r): i for i, r in enumerate(sorted_user_ids)}
        self.i_inx = {np.int64(r): i for i, r in enumerate(sorted_movie_ids)}

        # Xavier initialization for factors
        xavier_std = np.sqrt(2.0 / (len(self.u_inx) + len(self.i_inx)))
        self.item_factors = np.random.normal(0, xavier_std, (len(self.i_inx), k))
        self.user_factors = np.random.normal(0, xavier_std, (len(self.u_inx), k))

        return self

    def predict_batch(self, user_indices, item_indices, factor=None):
        """Vectorized prediction for multiple user-item pairs"""
        if factor is not None:
            # Use only up to factor+1 dimensions
            user_vecs = self.user_factors[user_indices, :factor+1]
            item_vecs = self.item_factors[item_indices, :factor+1]
            base_preds = np.sum(user_vecs * item_vecs, axis=1)
        else:
            # Use all dimensions
            base_preds = np.sum(
                self.user_factors[user_indices] * self.item_factors[item_indices],
                axis=1
            )
        
        user_biases = np.array([self.user_bias[int(u)] for u in user_indices])
        item_biases = np.array([self.item_bias[int(i)] for i in item_indices])
        
        predictions = self.avg_rating + base_preds + user_biases + item_biases
        return np.clip(predictions, 0.0, 5.0)

    def calculate_rmse(self, ratings, factor):
        """Vectorized RMSE calculation"""
        try:
            # Vectorized approach 
            user_indices = np.array([self.u_inx[uid] for uid, _, _ in ratings])
            item_indices = np.array([self.i_inx[iid] for _, iid, _ in ratings])
            actual_ratings = np.array([float(r) for _, _, r in ratings])

            # Get predictions for all ratings at once
            predictions = self.predict_batch(user_indices, item_indices)

            squared_errors = (predictions - actual_ratings) ** 2
            rmse = np.sqrt(np.mean(squared_errors))
            
            return rmse
        
        except Exception as e:
            print("\nError in calculate_rmse:")
            print("Error type:", type(e))
            print("Error message:", str(e))
            print("Ratings sample:", ratings[:5])
            raise

    def calculate_gds(self, ratings, index_random, factor):
        """Vectorized gradient calculation with batch processing"""
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

                # Update biases vectorized
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

    def train(self, rating_df: pd.DataFrame, k=20):
        """Train the model using vectorized operations"""
        print("Starting FunkSVD training...")
        
        # Ensure consistent type conversion
        rating_df['user_id'] = rating_df['user_id'].astype(np.int64)
        rating_df['movie_id'] = rating_df['movie_id'].astype(np.int64)
        
        # Initialize model factors
        self.initialize_factors(rating_df, k)
        
        # Prepare ratings array
        ratings = rating_df[['user_id', 'movie_id', 'rating']].values
        n_samples = len(ratings)
        
        # Generate randomized indices
        index_randomized = random.sample(range(0, n_samples), (n_samples - 1))

        # Iterate through factors
        for factor in range(k):
            fac_start_time = datetime.now()
            iter = 0
            best_rmse = float('inf')
            no_improvement = 0
            current_learning_rate = float(self.learning_rate)

            # Gradient descent for current factor
            while iter < self.max_iterations:
                cur_rmse = self.calculate_gds(ratings, index_randomized, factor)

                # Check for improvement
                if cur_rmse < best_rmse:
                    best_rmse = cur_rmse
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if no_improvement >= 2:
                        current_learning_rate *= 0.5
                        self.learning_rate = current_learning_rate
                        print(f"Reduce learning rate since there is no improvement to {current_learning_rate}")
                        no_improvement = 0

                # Convergence criteria
                if current_learning_rate < 1e-4 or (iter > 3 and cur_rmse > best_rmse):
                    print("Converged based on gradient norms")
                    break

                iter += 1
            
            # Print factor-level statistics
            print(f"Gradient descent for k = {factor} completed in {datetime.now() - fac_start_time}")
            print(f"Final RMSE: {best_rmse:.4f}")

        # Save the model
        self.save()
        print("Training completed")

    def save(self):
        """Save model parameters"""
        print(f"Saving model to {self.save_path}")
        
        # Save factors
        user_factors_df = pd.DataFrame(self.user_factors, index=list(self.user_ids))
        item_factors_df = pd.DataFrame(self.item_factors, index=list(self.movie_ids))
        
        # Save biases
        user_bias = {uid: self.user_bias[self.u_inx[uid]] 
                    for uid in self.u_inx.keys()}
        item_bias = {iid: self.item_bias[self.i_inx[iid]] 
                    for iid in self.i_inx.keys()}

        # Ensure save path exists
        os.makedirs(self.save_path, exist_ok=True)

        # Save all components
        with open(os.path.join(self.save_path, 'user_factors.json'), 'w') as f:
            f.write(user_factors_df.to_json())
        with open(os.path.join(self.save_path, 'item_factors.json'), 'w') as f:
            f.write(item_factors_df.to_json())
        with open(os.path.join(self.save_path, 'user_bias.data'), 'wb') as f:
            pickle.dump(user_bias, f)
        with open(os.path.join(self.save_path, 'item_bias.data'), 'wb') as f:
            pickle.dump(item_bias, f)
        with open(os.path.join(self.save_path, 'metadata.json'), 'w') as f:
            json.dump({
                'avg_rating': self.avg_rating,
                'n_users': len(self.user_ids),
                'n_items': len(self.movie_ids)
            }, f)
