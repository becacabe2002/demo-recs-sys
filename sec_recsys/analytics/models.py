from django.db import models


class Rating(models.Model):
    user_id = models.CharField(max_length=16)
    movie_id = models.CharField(max_length=16)
    enc_rating = models.BigIntegerField(default=0)
    rating_timestamp = models.DateTimeField()
    type = models.CharField(max_length=10, 
                            choices= [('real', 'Real'), ('fake', 'Fake')],
                            default='real')

    class Meta:
        indexes = [
            models.Index(fields=['user_id', 'movie_id']),
            models.Index(fields=['type'])
        ]
    def __str__(self):
        return "user_id: {}, movie_id: {}, rating: {}, type: {}"\
            .format(self.user_id, self.movie_id, self.enc_rating, self.type)


class Cluster(models.Model):
    cluster_id = models.IntegerField()
    user_id = models.CharField(max_length=16)

    def __str__(self):
        return "{} in {}".format(self.user_id, self.cluster_id)
