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
    for column in reader:
        movieCds.append(column['movieCd'])
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
    if type(response.get('movieInfoResult').get('movieInfo').get('audits')) == list and response.get('movieInfoResult').get('movieInfo').get('audits')[0]:
        movie['watchGradeNm'] = response.get('movieInfoResult').get('movieInfo').get('audits')[0].get('watchGradeNm')
    else:
        movie['watchGradeNm'] = response.get('movieInfoResult').get('movieInfo').get('audits').get('watchGradeNm')

    movie['openDt'] = response.get('movieInfoResult').get('movieInfo').get('openDt')
    movie['showTm'] = response.get('movieInfoResult').get('movieInfo').get('showTm')
    
    # genres list 여러개
    tmp = []
    if type(response.get('movieInfoResult').get('movieInfo').get('genres')) == list and response.get('movieInfoResult').get('movieInfo').get('genres')[0]:
        for genre in response.get('movieInfoResult').get('movieInfo').get('genres'):
            tmp.append(genre.get('genreNm'))
    movie['genreNm'] = ','.join(tmp)

    # director
    tmp = []
    if type(response.get('movieInfoResult').get('movieInfo').get('directors')) == list and response.get('movieInfoResult').get('movieInfo').get('directors')[0]:
        for director in response.get('movieInfoResult').get('movieInfo').get('directors'):
            tmp.append(director.get('peopleNm'))
    movie['peopleNm'] = ','.join(tmp)

    # directors = []
    # for i in range(len(response.get('movieInfoResult').get('movieInfo').get('directors'))):
    #     directors.append(response.get('movieInfoResult').get('movieInfo').get('directors')[i].get('peopleNm'))
    # movie['peopleNm'] = directors

    # 모든 영화 리스트에 영화 내용을 저장
    movies.append(movie)

# make file and write header
with open('movie_1.csv', 'w', encoding='utf-8', newline='') as f:
    fieldnames = ['movieCd', 'movieNm', 'movieNmEn', 'movieNmOg', 'watchGradeNm', 'openDt', 'showTm', 'genreNm', 'peopleNm']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

# data parsing
for movie in movies:
    with open('movie_2.csv', 'a', encoding='utf-8', newline='') as f:
        fieldnames = ['movieCd', 'movieNm', 'movieNmEn', 'movieNmOg', 'watchGradeNm', 'openDt', 'showTm', 'genreNm', 'peopleNm']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(movie)