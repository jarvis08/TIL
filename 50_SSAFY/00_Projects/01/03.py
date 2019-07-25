import requests
from decouple import config
import csv
from pprint import pprint
from datetime import date, timedelta


# peopleCd, peopleNm, repRoleNm, filmoNames
# 영화인 코드 , 영화인명 , 분야 , 필모리스트

kobis_token = config('KOBIS_KEY')
base = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/people/searchPeopleList.json'

# parsing director names from movie.csv
peopleNms = []
with open('movie.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for column in reader:
        director = column['peopleNm'].replace("'", '').replace('[', '').replace(']', '')
        if ',' in director:
            directors = director.split(',')
            for i in range(len(directors)):
                peopleNms.append(directors[i])
        else:
            peopleNms.append(director)
print(peopleNms)


# parsing director info with api
directors = []
for name in peopleNms:
    print(name)
    url = f'{base}?key={kobis_token}&peopleNm={name}'
    response = requests.get(url).json()
    #pprint(response)
    peopleCd = response.get('peopleListResult').get('peopleList')[0].get('peopleCd')
    peopleNm = name
    repRoleNm = response.get('peopleListResult').get('peopleList')[1].get('peopleCd')
    filmoNames = response.get('peopleListResult').get('peopleList')[0].get('peopleCd')
# people = []

# response.get('movieInfoResult').get('movieInfo').get('directors')[i].get('peopleNm')

# movies.append(movie)

# # make file and write header
# with open('director.csv', 'w', encoding='utf-8', newline='') as f:
#     fieldnames = ['movieCd', 'movieNm', 'movieNmEn', 'movieNmOg', 'watchGradeNm', 'openDt', 'showTm', 'genreNm', 'peopleNm']
#     writer = csv.DictWriter(f, fieldnames=fieldnames)
#     writer.writeheader()

# # data parsing
# for movie in movies:
#     with open('director.csv', 'a', encoding='utf-8', newline='') as f:
#         fieldnames = ['movieCd', 'movieNm', 'movieNmEn', 'movieNmOg', 'watchGradeNm', 'openDt', 'showTm', 'genreNm', 'peopleNm']
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writerow(movie)