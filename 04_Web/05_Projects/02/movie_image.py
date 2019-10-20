import requests
from pprint import pprint
import csv

image_links = []
movieCds = []
with open('movie_naver.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        movieCd, image_link = row['movieCd'], row['image']
        if len(image_link) < 3:
            continue
        image_links.append(image_link)
        movieCds.append(movieCd)

for i in range(len(image_links)):
    response = requests.get(image_links[i], stream=True)

    with open('./images/{}.png'.format(movieCds[i]), 'wb') as f:
        for chunk in response:
            f.write(chunk)