
docker compose run web python manage.py makemigrations --noinput
docker compose run web python manage.py migrate --run-syncdb

docker compose run web python populate_moviegeek.py
docker compose run web python populate_ratings.py
# docker compose run web python populate_sample_of_descriptions.py
docker compose run web python populate_logs.py
