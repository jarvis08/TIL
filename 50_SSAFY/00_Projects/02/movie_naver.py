import requests
from pprint import pprint
from decouple import config
import csv
from time import sleep

movieCds = []
movieNms = []
openDts = []
peopleNms = []
with open('./RFP/movie.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        movieCd, movieNm, openDt, peopleNm = row['영화 대표코드'], row['영화명(국문)'], row['개봉연도'], row['감독']
        movieCds.append(movieCd)
        movieNms.append(movieNm)
        openDts.append(openDt)
        peopleNms.append(peopleNm)

# Naver API에 보낼 url 만들기
config('KOBIS_KEY')
base = 	'https://openapi.naver.com/v1/search/movie.json'
headers = {
    'X-Naver-Client-Id': config('CLIENT_ID'),
    'X-Naver-Client-Secret': config('CLIENT_SECRET')
}

# 영화 별 정보 dictionary를 list에 담기
movies = []
for i in range(len(movieNms)):
    movie = dict()

    query = movieNms[i]
    url = f'{base}?query={query}'
    response = requests.get(url, headers=headers).json()
    res = response.get('items')
    movie['movieCd'] = movieCds[i]
    found = False

    # 영화 제목 검색 시 여러 편의 영화가 포함되므로, 가장 적절한 영화 탐색
    # 1. 개봉 연도, 감독명이 같은 경우를 탐색
    for l in range(len(res)):
        if openDts[i][:4] == res[l].get('pubDate') and peopleNms[i] in res[l].get('director'):
            movie['link'] = res[l].get('link')
            movie['image'] = res[l].get('image')
            movie['userRating'] = response.get('items')[l].get('userRating')
            found = True
            movies.append(movie)
            break
    # 2. 앞에서 못 찾을 경우, 개봉 연도가 같은 경우로 조건 축소
    if not found:
        for l in range(len(res)):
            if openDts[i] == res[l].get('pubDate'):
                movie['link'] = res[l].get('link')
                movie['image'] = res[l].get('image')
                movie['userRating'] = response.get('items')[l].get('userRating')
                found = True
                movies.append(movie)
                break
    # 3. 그래도 못 찾을 경우, 0번째 할당
    if not found:
        if len(res[0].get('link')) > 3:
            movie['link'] = res[0].get('link')
        if len(response.get('items')[0].get('image')) > 3:
            movie['image'] = res[0].get('image')
            movie['userRating'] = res[0].get('userRating')
        movies.append(movie)
    sleep(0.05)


with open('movie_naver.csv', 'w', encoding='UTF-8', newline='') as f:
    fieldnames = ['movieCd', 'link', 'image', 'userRating']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

# data parsing
for movie in movies:
    with open('movie_naver.csv', 'a', encoding='utf-8', newline='') as f:
        fieldnames = ['movieCd', 'link', 'image', 'userRating']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(movie)