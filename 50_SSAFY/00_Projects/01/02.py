import requests
from decouple import config
import csv
from pprint import pprint
from datetime import date, timedelta


# movieCd, movieNm, movieNmEn, movieNmOg, watchGradeNm, openDt, showTm, genreNm, peopleNm
# 영화 대표코드 , 영화명(국문) , 영화명(영문) , 영화명(원문) , 관람등급 , 개봉연도 , 상영시간 , 장르 , 감독명

kobis_token = config('KOBIS_KEY')
base = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieInfo.json'

# parsing movie info from boxoffice.csv
movieCds = []
with open('boxoffice.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        movieCds.append(row['movieCd'])
# movieCds = ['20196309', '20183867']
movies = []

for movieCd in movieCds:
    movie = dict()
    url = f'{base}?key={kobis_token}&movieCd={movieCd}'
    response = requests.get(url).json()
    # pprint(response)
    movie['movieCd'] = movieCd
    movie['movieNm'] = response.get('movieInfoResult').get('movieInfo').get('movieNm')
    movie['movieNmEn'] = response.get('movieInfoResult').get('movieInfo').get('movieNmEn')
    movie['movieNmOg'] = response.get('movieInfoResult').get('movieInfo').get('movieNmOg')
    # watchGradeNm list 여러개
    gradeNm = []
    for i in range(len(response.get('movieInfoResult').get('movieInfo').get('audits'))):
         gradeNm.append(response.get('movieInfoResult').get('movieInfo').get('audits')[i].get('watchGradeNm'))

    movie['watchGradeNm'] = gradeNm
    movie['openDt'] = response.get('movieInfoResult').get('movieInfo').get('openDt')
    movie['showTm'] = response.get('movieInfoResult').get('movieInfo').get('showTm')
    
    # genres list 여러개
    genre = []
    for i in range(len(response.get('movieInfoResult').get('movieInfo').get('genres'))):
        genre.append(response.get('movieInfoResult').get('movieInfo').get('genres')[i].get('genreNm'))
    movie['genreNm'] = genre

    # director
    directors = []
    for i in range(len(response.get('movieInfoResult').get('movieInfo').get('directors'))):
        directors.append(response.get('movieInfoResult').get('movieInfo').get('directors')[i].get('peopleNm'))
    movie['peopleNm'] = directors

    movies.append(movie)

# make file and write header
with open('movie.csv', 'w', encoding='utf-8', newline='') as f:
    fieldnames = ['movieCd', 'movieNm', 'movieNmEn', 'movieNmOg', 'watchGradeNm', 'openDt', 'showTm', 'genreNm', 'peopleNm']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

# data parsing
for movie in movies:
    with open('movie.csv', 'a', encoding='utf-8', newline='') as f:
        fieldnames = ['movieCd', 'movieNm', 'movieNmEn', 'movieNmOg', 'watchGradeNm', 'openDt', 'showTm', 'genreNm', 'peopleNm']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(movie)