from django.db import models

# Create your models here.
class Movie(models.Model):
    title = models.CharField(max_length=40)
    title_en = models.CharField(max_length=50)
    audience = models.IntegerField()
    open_date = models.DateField()
    genre = models.CharField(max_length=30)
    watch_grade = models.CharField(max_length=40)
    score = models.FloatField()
    poster_url = models.TextField()
    description = models.TextField()