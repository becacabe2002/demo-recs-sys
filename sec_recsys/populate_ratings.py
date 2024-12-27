import os
import urllib.request
import django
import datetime
import decimal
from tqdm import tqdm
import random
import argparse
import numpy as np
import requests
from django.conf import settings
from utils.user_enc import UserEnc

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'prs_project.settings')

django.setup()

from analytics.models import Rating

def download_ratings():
    URL = 'https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat'
    response = urllib.request.urlopen(URL)
    data = response.read()

    print('download finished')
    return data.decode('utf-8')


def delete_db():
    print('truncate db')
    Rating.objects.all().delete()
    print('finished truncate db')

def process_rating(user_enc, ratings, fake_ratio=0.1):
    reals = [
        {
            'user_id': r[0],
            'item_id': r[1],
            'rating': float(r[2]),
            'is_real': True
            }
             for r in ratings
        ]
    enc_ratings = user_enc.inject_fake_ratings(reals, fake_ratio)

    resp = requests.post(
            f"{settings.RECSYS_URL}/analytics/api/secure-ratings/upload_enc/",
            json={'ratings': enc_ratings}
            )
    return resp.status_code == 200

def populate(sample_size=None, fake_ratio=0.1):

    delete_db()
    
    print("Initialize encryption\n")
    user_enc = UserEnc()

    raw_data = download_ratings()
    ratings = [r.split(sep="::") for r in raw_data.split(sep="\n") if r]
    if sample_size:
        ratings = random.sample(ratings, sample_size)
    
    process_rating(user_enc, ratings, fake_ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, help='Number of sample ratings')
    parser.add_argument('--fake_ratio', type=float, help='Ratio of fake/real ratings')
    args = parser.parse_args()
    print("Starting MovieGeeks Population script...")
    populate(args.sample, args.fake_ratio)
