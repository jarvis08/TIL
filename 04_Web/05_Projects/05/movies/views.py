from django.shortcuts import render, redirect
from .models import Movie

# Create your views here.
def index(request):
    movies = Movie.objects.all()
    context = {
        'movies': reversed(movies),
    }
    return render(request, 'movies/index.html', context)


def new(request):
    return render(request, 'movies/new.html')


def create(request):
    movie = Movie()
    movie.title = request.GET.get('title')
    movie.title_en = request.GET.get('title_en')
    movie.audience = request.GET.get('audience')
    movie.open_date = request.GET.get('open_date')
    movie.genre = request.GET.get('genre')
    movie.watch_grade = request.GET.get('watch_grade')
    movie.score = request.GET.get('score')
    movie.poster_url = request.GET.get('poster_url')
    movie.description = request.GET.get('description')
    movie.save()
    movie = Movie.objects.get(title=request.GET.get('title'))
    return redirect('movies:detail', movie.pk)


def detail(request, pk):
    movie = Movie.objects.get(pk=pk)
    context = {
        'movie': movie,
    }
    return render(request, 'movies/detail.html', context)


def edit(request, pk):
    movie = Movie.objects.get(pk=pk)
    context = {
        'movie': movie,
    }
    return render(request, 'movies/edit.html', context)


def update(request, pk):
    movie = Movie.objects.get(pk=pk)
    movie.title = request.GET.get('title')
    movie.title_en = request.GET.get('title_en')
    movie.audience = request.GET.get('audience')
    movie.open_date = request.GET.get('open_date')
    movie.genre = request.GET.get('genre')
    movie.watch_grade = request.GET.get('watch_grade')
    movie.score = request.GET.get('score')
    movie.poster_url = request.GET.get('poster_url')
    movie.description = request.GET.get('description')
    movie.save()
    return redirect('movies:detail', pk)


def delete(request, pk):
    movie = Movie.objects.get(pk=pk)
    movie.delete()
    return redirect('home')