import requests
from decouple import config
import csv
from pprint import pprint
from datetime import date, timedelta

def get_Dts(last_date, num_weeks):
    Dts = [last_date]
    weeks = num_weeks - 1
    while True:
        # format : 2019-07-01
        Dt = date.fromisoformat(Dts[-1]) - timedelta(days=7)
        Dts.append(Dt.isoformat())
        weeks -= 1
        if not weeks:
            break
    return Dts

num_weeks = 50
kobis_token = config('KOBIS_KEY')
weekGb = '0'
base = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchWeeklyBoxOfficeList.json'
targetDt = get_Dts('2019-07-13', num_weeks)
for Dt in targetDt:
    cleaned = Dt.replace('-', '')
    targetDt[targetDt.index(Dt)] = cleaned

total = []
for Dt in targetDt:
    url = f'{base}?key={kobis_token}&targetDt={Dt}&weekGb={weekGb}'
    response = requests.get(url).json()
    # pprint(response)
    for i in range(10):
        # movieCd, movieNm, audiAcc
        movieCd = response.get('boxOfficeResult').get('weeklyBoxOfficeList')[i].get('movieCd')
        movieNm = response.get('boxOfficeResult').get('weeklyBoxOfficeList')[i].get('movieNm')
        audiAcc = response.get('boxOfficeResult').get('weeklyBoxOfficeList')[i].get('audiAcc')
        for movie in total:
            if movie.get('movieCd') == movieCd:
                break
        else:
            weekly = dict()
            weekly['movieCd'] = movieCd
            weekly['movieNm'] = movieNm
            weekly['audiAcc'] = audiAcc
            total.append(weekly)


# make file and write header
with open('boxoffice.csv', 'w', encoding='utf-8', newline='') as f:
    fieldnames = ['movieCd', 'movieNm', 'audiAcc']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

# data parsing
for dic in total:
    with open('boxoffice.csv', 'a', encoding='utf-8', newline='') as f:
        fieldnames = ['movieCd', 'movieNm', 'audiAcc']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # writer.writeheader()
        writer.writerow(dic)