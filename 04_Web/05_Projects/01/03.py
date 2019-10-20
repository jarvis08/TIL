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
    for row in reader:
        director = row['peopleNm'].replace("'", '').replace('[', '').replace(']', '')
        if ',' in director:
            directors = director.split(',')
            for i in range(len(directors)):
                peopleNms.append(directors[i])
        else:
            peopleNms.append(director)

# parsing director info with api
directors = []
for name in peopleNms:
    url = f'{base}?key={kobis_token}&peopleNm={name}'
    response = requests.get(url).json()
    # pprint(response)
    # for문 하나에 peopleCd, rep, filmo 포함
    for i in range(len(response.get('peopleListResult').get('peopleList'))):
        director = dict()
        director['peopleCd'] = response.get('peopleListResult').get('peopleList')[i].get('peopleCd')
        director['peopleNm'] = name
        director['repRoleNm'] = response.get('peopleListResult').get('peopleList')[i].get('repRoleNm')
        director['filmoNames'] = response.get('peopleListResult').get('peopleList')[i].get('filmoNames')
        directors.append(director)

# make file and write header
with open('director.csv', 'w', encoding='utf-8', newline='') as f:
    fieldnames = ['peopleCd', 'peopleNm', 'repRoleNm', 'filmoNames']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

# data parsing
for director in directors:
    with open('director.csv', 'a', encoding='utf-8', newline='') as f:
        fieldnames = ['peopleCd', 'peopleNm', 'repRoleNm', 'filmoNames']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(director)