# search_engine_ranking.py
import requests
import bs4

# 여러개 긁어오기
## select_one
naver_url = "https://www.naver.com/"
response = requests.get(naver_url).text
document = bs4.BeautifulSoup(response, 'html.parser')
for i in range(1, 11):
    searched = document.select_one('ul.ah_l:nth-child(5) > li:nth-child({}) > a:nth-child(1) > span:nth-child(2)'.format(i)).text
    print("네이버 {}위 검색어.\t".format(i) + searched)

## select class=ah_k
ranked = document.select('.ah_k', limit=10)
i = 1
for item in ranked:
    print("{}.\t".format(i), item.text)
    i += 1