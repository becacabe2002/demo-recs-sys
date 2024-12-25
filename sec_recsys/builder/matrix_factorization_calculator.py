import logging
import os
import random
import sys
import pickle

import numpy as np
import pandas as pd
from decimal import Decimal
from collections import defaultdict
import math
from datetime import datetime

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "prs_project.settings")

import django


django.setup()

from analytics.models import Rating


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


class MatrixFactorization(object):

    def __init__(self, save_path, learning_rate=0.002, lambda_reg=0.002, mu_reg=0.002, max_iterations=10):
        self.logger = logging.getLogger('BaseMF')
        self.save_path = save_path
        self.learning_rate = Decimal(learning_rate)
        self.lambda_reg = Decimal(lambda_reg) # regularization param for user factor gd
        self.mu_reg = Decimal(mu_reg) # regularization param for item factors gd
        self.MAX_ITERATIONS = max_iterations

        self.user_factors = None
        self.item_factors = None
        self.item_counts = None
        self.u_inx = None
        self.i_inx = None
        self.user_ids = None
        self.movie_ids = None

        self.number_of_ratings = 0
        random.seed(42)

        ensure_dir(save_path)

    def initialize_factors(self, ratings, k=25):
        self.user_ids = set(ratings['user_id'].unique())
        self.movie_ids = set(ratings['movie_id'].unique())
        print(f"Number of user: {len(self.user_ids)} - Number of item: {len(self.movie_ids)}")
        self.item_counts = ratings[['movie_id', 'rating']].groupby('movie_id').count()
        self.item_counts = self.item_counts.reset_index()

        self.item_sum = ratings[['movie_id', 'rating']].groupby('movie_id').sum()
        self.item_sum = self.item_sum.reset_index()

        self.u_inx = {r: i for i, r in enumerate(self.user_ids)}
        self.i_inx = {r: i for i, r in enumerate(self.movie_ids)}

        self.item_factors = np.random.normal(0, 0.1, (len(self.i_inx), k))
        self.user_factors = np.random.normal(0, 0.1, (len(self.u_inx), k))

        self.logger.info("user_factors are {}".format(self.user_factors.shape))
    
    def normalize_factors(self):
        self.logger.info("Normalizing factors")
        user_norms = np.linalg.norm(self.user_factors, axis=1)
        item_norms = np.linalg.norm(self.item_factors, axis=1)
        
        # Avoid division by zero
        user_norms[user_norms == 0] = 1
        item_norms[item_norms == 0] = 1
        
        self.user_factors /= user_norms[:, np.newaxis]
        self.item_factors /= item_norms[:, np.newaxis]

    def predict(self, user, item): 
        p = np.dot(self.item_factors[item], self.user_factors[user])
        prediction = Decimal(str(p))
        return prediction
    
    def build(self, ratings, params=None): 
        if params:
            k = params['k']
            self.save_path = params['save_path']
        else:
            k = 10
        self.train(ratings, k) 
    
    def calculate_gds(self, ratings):

        user_gradients = np.zeros_like(self.user_factors, dtype=np.float64)
        item_gradients = np.zeros_like(self.item_factors, dtype=np.float64)
        
        user_counts = np.zeros(len(self.u_inx))
        item_counts = np.zeros(len(self.i_inx))

        total_square_err = 0.0
        for user_id, item_id, rating in ratings:            
            u = self.u_inx[user_id]
            i = self.i_inx[item_id]
            # Calculate error (rij - ⟨ui,vj⟩) 
            predict = self.predict(u, i)
            error = float(predict - Decimal(rating))
            total_square_err += error * error
    
            # Update gradients
            user_gradients[u] += self.item_factors[i] * error + float(self.lambda_reg) * self.user_factors[u]
            item_gradients[i] += self.user_factors[u] * error + float(self.mu_reg) * self.item_factors[i]
        
        rmse = np.sqrt(total_square_err / len(ratings))
        return user_gradients, item_gradients, rmse
    
    def calculate_gradient_norms(self, ugd, igd):
        '''
        Calculate norm as stopping criteria
        '''
        user_norm = np.linalg.norm(ugd)
        item_norm = np.linalg.norm(igd)
        return user_norm, item_norm
    
    def train(self, rating_df, k=25):
        self.initialize_factors(rating_df, k)
        self.logger.info(f"Start training at {datetime.now()}")
        
        ratings = rating_df[['user_id', 'movie_id', 'rating']].values
        iter = 0
        current_learning_rate = float(self.learning_rate)
        best_rmse = float('inf')
        no_improvement = 0

        while iter < self.MAX_ITERATIONS:
            start_time = datetime.now()
            user_gds, item_gds, cur_rmse = self.calculate_gds(ratings)
            
            # Update gds
            self.user_factors -= float(current_learning_rate) * user_gds
            self.item_factors -= float(current_learning_rate) * item_gds
            
            # self.normalize_factors()

            user_norm, item_norm = self.calculate_gradient_norms(user_gds, item_gds)
            self.logger.info(f"Iteration {iter + 1} completed in {datetime.now() - start_time}")
            self.logger.info(f"Gradient norms - User: {user_norm}, Item: {item_norm}")
            self.logger.info(f"RMSE: {cur_rmse:.4f}")
            
            if cur_rmse < best_rmse:
                best_rmse = cur_rmse
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= 2:
                  current_learning_rate *= 0.5
                  no_improvement = 0
                  self.logger.info(f"Reduce learning rate since there is no improvement to {current_learning_rate}")
            # Stop if gradients are small enough
            if current_learning_rate < 1e-5  or (iter > 3 and cur_rmse > best_rmse):
                self.logger.info("Converged based on gradient norms")
                break
                
            iter += 1
        self.save()

    def save(self): 
        ensure_dir(self.save_path)

        self.logger.info("saving factors in {}".format(self.save_path))
        uf = pd.DataFrame(self.user_factors, index=self.user_ids)
        it_f = pd.DataFrame(self.item_factors, index=self.movie_ids)

        with open(self.save_path + 'user_factors.json', 'w') as outfile:
            outfile.write(uf.to_json())
        with open(self.save_path + 'item_factors.json', 'w') as outfile:
            outfile.write(it_f.to_json()) 

def load_all_ratings(min_ratings=1):
    columns = ['user_id', 'movie_id', 'rating', 'type', 'rating_timestamp']

    ratings_data = Rating.objects.all().values(*columns)
    ratings = pd.DataFrame.from_records(ratings_data, columns=columns)

    user_count = ratings[['user_id', 'movie_id']].groupby('user_id').count()
    user_count = user_count.reset_index()
    user_ids = user_count[user_count['movie_id'] > min_ratings]['user_id']
    ratings = ratings[ratings['user_id'].isin(user_ids)]

    ratings['rating'] = ratings['rating'].apply(Decimal)
    return ratings


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger('BaseMF')
    logger.info("[BEGIN] Calculating matrix factorization")

    MF = MatrixFactorization(save_path='./models/baseMF/', max_iterations=20)
    loaded_ratings = load_all_ratings(20)
    logger.info("using {} ratings".format(loaded_ratings.shape[0]))
    MF.train(load_all_ratings(), k=20)
    logger.info("[DONE] Calculating matrix factorization")
