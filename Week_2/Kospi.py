# kospi.py
import requests
import bs4

finance_url = "https://finance.naver.com/sise/"

########################################
# response = requests.get(finance_url)
# response = requests.get(finance_url).status_code
# print(response)
"""Result
$ python kospi.py
<Response [200]>
"""
# 200의 경우 양호한 Status Code
# F12 > Network 탭 > Name(www.daum.net) 선택 > Headers 탭 > General 탭 > Status Code
# Google Image 검색 후, Status Code 참조
# 400대의 경우 주로 코드 문제
########################################

response = requests.get(finance_url).text
# .text 시 페이지 소스 보기 결과 중 맨 하단 부분과 동일한 내용

document = bs4.BeautifulSoup(response, 'html.parser')
# bs4의 BeautifulSoup 함수를 이용해 response를 python이 보기 쉽도록 변형
# 사람의 눈에는 크게 다르지 않지만, 식별자(id)를 기준으로 검색이 가능해짐
# , 'html.parser'를 지정해주지 않으면 parser 미지정으로 경고

# print(document.select_one('#KOSPI_now'))
"""Result
$ python kospi.py
<span class="num num2" id="KOSPI_now">2,063.12</span>
"""
# print(document.select_one('#KOSPI_now').text)
"""
$ python kospi.py
2,061.84
"""

kospi = document.select_one('#KOSPI_now').text
kosdaq = document.select_one('#KOSDAQ_now').text
kospi_200 = document.select_one('#KPI200_now').text
# Web에서 원하는 부분의 경로 복사하기
# id의 경우 #을 사용하며, class는 .
"""
Web 상의 원하는 부분에 (마우스 우클릭 > 요소 검사) 실행 시 원하는 부분의 코드를 바로 찾아줌
찾아진 해당 라인에 (마우스 우클릭 > 복사 > CSS 선택자) 선택
naver의 경우 (마우스 우클릭 > copy > copy selector) 선택
"""

print('현재 코스피 지수 :: ' + kospi)
"""
$ python kospi.py
현재 코스피 지수 = 2,060.80
"""