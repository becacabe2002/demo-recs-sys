import os
import django
import json
import requests
import time
from datetime import datetime, timedelta

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'prs_project.settings')

django.setup()

from recommender.models import MovieDescriptions


# Start date and URL for TMDb API
START_DATE = "2012-03-01"  # Adjust as needed
END_DATE = "2023-12-31"  # Adjust as needed
MAX_PAGES = 500  # TMDb API limit

def get_descriptions():
    api_key = get_api_key()
    start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(END_DATE, "%Y-%m-%d")

    while start_date < end_date:
        chunk_start_date = start_date
        chunk_end_date = min(chunk_start_date + timedelta(days=30), end_date)

        print(f"Fetching movies from {chunk_start_date.date()} to {chunk_end_date.date()}")

        for page in range(1, MAX_PAGES + 1):
            url = f"https://api.themoviedb.org/3/discover/movie?primary_release_date.gte={chunk_start_date.date()}&primary_release_date.lte={chunk_end_date.date()}&api_key={api_key}&page={page}"
            print(f"Fetching page {page}: {url}")

            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                break

            data = response.json()
            if 'results' not in data or not data['results']:
                print(f"No more results on page {page}.")
                break

            for film in data['results']:
                id = film['id']
                md, created = MovieDescriptions.objects.get_or_create(movie_id=id)

                md.imdb_id = get_imdb_id(id)
                md.title = film['title']
                md.description = film['overview']
                md.genres = film['genre_ids']

                if md.imdb_id:
                    md.save()

            time.sleep(0.3)  # Avoid hitting API rate limits

        # Move to the next date range
        start_date = chunk_end_date + timedelta(days=1)
        print(f"Completed chunk: {chunk_start_date.date()} to {chunk_end_date.date()}")

def get_imdb_id(moviedb_id):
    url = """https://api.themoviedb.org/3/movie/{}?api_key={}"""

    r = requests.get(url.format(moviedb_id, get_api_key()))

    json = r.json()
    print(json)
    if 'imdb_id' not in json:
        return ''
    imdb_id = json['imdb_id']
    if imdb_id is not None:
        print(imdb_id)
        return imdb_id[2:]
    else:
        return ''


def get_api_key():
    # Load credentials
    cred = json.loads(open(".prs").read())
    return cred['themoviedb_apikey']


def get_popular_films_for_genre(genre_str):
    film_genres = {'drama': 18, 'action': 28, 'comedy': 35}
    genre = film_genres[genre_str]

    url = """https://api.themoviedb.org/3/discover/movie?sort_by=popularity.desc&with_genres={}&api_key={}"""
    api_key = get_api_key()
    r = requests.get(url.format(genre, api_key))
    print(r.json())
    films = []
    for film in r.json()['results']:
        id = film['id']
        imdb = get_imdb_id(id)
        print("{} {}".format(imdb, film['title']))
        films.append(imdb[2:])
    print(films)


if __name__ == '__main__':
    print("Starting MovieGeeks Population script...")
    get_descriptions()
    # get_popular_films_for_genre('comedy')
