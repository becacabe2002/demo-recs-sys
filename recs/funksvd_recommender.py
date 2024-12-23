import collections
import decimal
import json
import pickle
from decimal import Decimal

import pandas as pd
from django.db.models import Avg

from analytics.models import Rating
from recommender.models import Recs
from recs.base_recommender import base_recommender


class FunkSVDRecs(base_recommender):

    def __init__(self, save_path='./models/funkSVD/'):
        self.save_path = save_path
        self.model_loaded = False
        self.avg = Decimal(list(Rating.objects.all().aggregate(Avg('rating')).values())[0])

    def load_model(self, save_path):
        with open(save_path + 'user_bias.data', 'rb') as ub_file:
            self.user_bias = pickle.load(ub_file)
        with open(save_path + 'item_bias.data', 'rb') as ub_file:
            self.item_bias = pickle.load(ub_file)
        with open(save_path + 'user_factors.json', 'r') as infile:
            self.user_factors = pd.DataFrame(json.load(infile)).T
        with open(save_path + 'item_factors.json', 'r') as infile:
            self.item_factors = pd.DataFrame(json.load(infile)).T
        self.ordered_item_bias = list(collections.OrderedDict(sorted(self.item_bias.items())).values())
        self.model_loaded = True

    def set_save_path(self, save_path):
        self.save_path = save_path

        self.load_model(save_path)

    def predict_score(self, user_id, item_id):

        if not self.model_loaded:
            self.load_model(self.save_path)

        rec = Recs.objects.filter(user_id = user_id, item_id=item_id).first()
        if rec is None:
            return 0
        else:
            return rec.rating

    def predict_score_by_rating(self, item_id, active_user_items):
        """
        Predict the score for a given movie (item_id) based on the active user's rating history.
        Uses collaborative filtering approach to estimate the score.
        """

        if not self.model_loaded:
            self.load_model(self.save_path)

        rated_movies = {}
        
        if isinstance(active_user_items, list) and isinstance(active_user_items[0], dict):
            rated_movies = {movie['movie_id']: movie['rating'] for movie in active_user_items}
        elif hasattr(active_user_items, 'values'):  # For Django querysets
            rated_movies = {movie['movie_id']: movie['rating'] for movie in active_user_items.values('movie_id', 'rating')}
        else:
            print("Unexpected structure for active_user_items:", active_user_items)
        
        top = Decimal(0.0)
        bottom = Decimal(0.0)

        if str(self.user_factors.columns) in active_user_items:
            user = self.user_factors[str(item_id)]

        for movie_id, rating in rated_movies.items():
            sim_item = self.item_factors.T.dot(user)
            
            similarity = sim_item.get(movie_id, Decimal(0.0))
            
            if similarity != Decimal(0):
                top += similarity * rating
                bottom += similarity

        if bottom == Decimal(0):
            return 0

        predicted_score = Decimal(top / bottom) + self.avg
        return predicted_score

    def recommend_items(self, user_id, num=6):

        if not self.model_loaded:
            self.load_model(self.save_path)

        active_user_items = Rating.objects.filter(user_id=user_id).order_by('-rating')[:100]

        return self.recommend_items_by_ratings(user_id, active_user_items.values())

    def recommend_items_by_ratings(self, user_id, active_user_items, num=6):

        if not self.model_loaded:
            self.load_model(self.save_path)

        rated_movies = {movie['movie_id']: movie['rating'] for movie in active_user_items}
        recs = {}
        if str(user_id) in self.user_factors.columns:

            user = self.user_factors[str(user_id)]

            scores = self.item_factors.T.dot(user)

            sorted_scores = scores.sort_values(ascending=False)
            result = sorted_scores[:num + len(rated_movies)].apply(decimal.Decimal)
            user_bias = 0

            if user_id in self.user_bias.keys():
                user_bias = self.user_bias[user_id]
            elif int(user_id) in self.user_bias.keys():
                user_bias = self.user_bias[int(user_id)]
                print(f'it was an int {user_bias}')

            rating = float(user_bias + self.avg)
            result += Decimal(rating)

            recs = {r[0]: {'prediction': r[1] + Decimal(float(self.item_bias[r[0]]))}
                    for r in zip(result.index, result) if r[0] not in rated_movies}

        sorted_items = sorted(recs.items(), key=lambda item: -float(item[1]['prediction']))[:num]

        return sorted_items
