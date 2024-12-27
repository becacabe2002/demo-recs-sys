from typing import Dict
import requests
import numpy as np
from decimal import Decimal
from django.conf import settings
from Cryptodome.PublicKey import ElGamal
from Cryptodome.Random import get_random_bytes
import base64
import random

class UserEnc:
    def __init__(self):
        self.csp_url = settings.CSP_URL
        self.ahe_pkey = None
        self.init_keys()
        
    def init_keys(self):
        resp = requests.post(f"{self.csp_url}/init")
        keys = resp.json()['public_keys']
        key_data = base64.b64decode(keys['ahe_pkey'])
        self.ahe_pkey = ElGamal.construct((int.from_bytes(key_data[:128], 'big'), 
                                            int.from_bytes(key_data[128:256], 'big'),
                                            int.from_bytes(key_data[256:], 'big'))) 

    def enc_rating(self, user_id, item_id, rating, is_real):
        '''
        Encrypt single rating using (partially) AHE
        '''
        if not self.ahe_pkey:
            raise ValueError("Not initialize ahe key")
        
        rating_int = int(rating * 100)
        k = get_random_bytes(16)
        enc_rating = self.ahe_pkey.encrypt(rating_int, k)

        indicator = 1 if is_real else 0
        k2 = get_random_bytes(16)
        enc_indicator = self.ahe_pkey.encrypt(enc, k)
        return {
            'user_id': user_id,
            'item_id': item_id,
            'rating': base64.b64encode(enc_rating[0].to_bytes(128, 'big')).decode(),
            'indicator': base64.b64encode(enc_indicator[0].to_bytes(128, 'big')).decode()
            }

    def inject_fake_ratings(self, reals, ratio):
        num_fakes = int(len(reals) * ratio)
        fakes = []
        for i in range(num_fakes):
            fakes.append({
                'user_id': reals[i]['user_id'],
                'item_id': str(random.randint(1, 10000)),
                'rating': random.uniform(1.0, 5.0),
                'is_real': False
                })
        all_ratings = reals + fakes
        random.shuffle(all_ratings)
        return [self.enc_rating(**rating) for rating in all_ratings]
                
