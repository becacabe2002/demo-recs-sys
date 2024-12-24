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


class BaseMFRecs(base_recommender):

    def __init__(self, save_path='./models/baseMF/'):
        self.save_path = save_path
        self.model_loaded = False
        self.avg = Decimal(list(Rating.objects.all().aggregate(Avg('rating')).values())[0])

    def load_model(self, save_path='./models/baseMF'):
        with open(save_path + 'user_factors.json', 'r') as infile:
            self.user_factors = pd.DataFrame(json.load(infile)).T
        with open(save_path + 'item_factors.json', 'r') as infile:
            self.item_factors = pd.DataFrame(json.load(infile)).T
        self.model_loaded = True

    def set_save_path(self, save_path):
        self.save_path = save_path

        self.load_model(save_path)

    def predict_score(self, user_id, item_id):
        if not self.model_loaded:
            self.load_model(self.save_path)

        rec = Recs.objects.filter(user_id=user_id, item_id=item_id).first()
        if rec is None:
            if str(user_id) not in self.user_factors.columns or \
               str(item_id) not in self.item_factors.columns:
                return 0
            
            prediction = self._calculate_prediction(user_id, item_id)
            return prediction
        else:
            return rec.rating

    def predict_score_by_ratings(self, item_id, user_ratings):
        if self.model_loaded == False:
            self.load_model()

        if not user_ratings or item_id not in self.item_factors.columns:
            return 0

        known_movie_id = next(iter(user_ratings.keys()))
        rating_entry = Rating.objects.filter(movie_id=known_movie_id).first()
        if not rating_entry:
            return 0
            
        user_id = rating_entry.user_id

        if user_id not in self.user_factors.columns:
            return 0
        prediction = self._calculate_prediction(user_id, item_id)
        return prediction

    def _calculate_prediction(self, user_id, movie_id):
       user = self.user_factors[str(user_id)]
       item = self.item_factors[str(movie_id)]
       prediction = float(user.dot(item))
       # Clamp prediction between 1 and 5
       prediction = max(1.0, min(5.0, prediction))
       return prediction 

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
        if str(user_id) not in self.user_factors.columns:
            return recs
        
        user = self.user_factors[str(user_id)]

        scores = self.item_factors.T.dot(user)
        sorted_scores = scores.sort_values(ascending=False)
        recommendations = []
        
        for item_id, score in sorted_scores.items():
            if item_id not in rated_movies and len(recommendations) < num:
                recommendations.append(
                    (item_id, {'prediction': float(score)})
                )
        return recommendations[:num]
