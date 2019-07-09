# SSAFY_Day_2

## Markdown Language

> `- + Enter`: black point
> `(```python + Enter)`: 블록 코드 강조 (back tick)
>
> ```python
> (Ctrl + Enter) : 빠져나가기
> ```
>
> `--- + Enter`: 줄긋기
> `Ctrl + T`: 표 만들기
> `back tick + 내용 + back tick`: 코드블록에 내용 넣기
> `> ` 단 띄우기

---

## CLI(Command Line Interface)

: 유닉스 shell(Linux, Git Bash, ...)
: CMD, Powershell

> `ls` : list 목록
> `cd`: 지정 위치로 이동
> `pwd`: 현위치 (point working directory)
> `mkdir`: 폴더 생성
> `code .`: vs code 현위치에서 실행
> `touch 문서.txt`: 문서를 생성
> `rm 문서.txt`: 문서 삭제

---

## Git

- 생활코딩 git 강의 좋다

```shell
cd (master directory)
git init
git add .
git commit -m "1"
# -m :: message와 함께
"""
Run

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"
"""
git config --global user.email "cdb921226@gmail.com"
git config --global user.name "jarvis08"
git remote add origin https://github.com/jarvis08/SSAFY.git
git push -u origin master

git rm -r 파일/디렉토리명
```

---

## PATH

- Shell이 설정된 Path에 해당 프로그램의 존재 여부 확인
  존재 시, 어느 위치에서도 해당 프로그램을 사용할 수 있도록 도움
- Windows의 경우 User보다 System 계정의 Path를 먼저 고려(Override)
  Linux의 경우 import된 순서로 Override
- 권한에 있어서,
  Windows는 '관리자 권한으로 실행'을 해야하며,
  Linux의 경우 Sudo 입력을 통해 관리자 Override가 가능

---

## Web

- **Client ---request---> Server**
request로 url을 사용
  
- **Server ---response---> Client**
  response로 문서를 받음
  문서 : 페이지에서 마우스 우클릭, 페이지 소스보기로 확인 가능

- 우클릭 후 검사, F12 등으로 확인 가능

- Web 활용하기
  1. Request 보내기
  2. Response인 문서를 저장하기
  3. 문서에서 필요한 내용 추출하기
  
  `pip install requests`
  `pip install bs4`

```python
# Kospi.py
import requests
import bs4

finance_url = "https://finance.naver.com/sise/"

########################################
# response = requests.get(finance_url)
# response = requests.get(finance_url).status_code
# print(response)
"""Result
$ python Kospi.py
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
$ python Kospi.py
<span class="num num2" id="KOSPI_now">2,063.12</span>
"""
# print(document.select_one('#KOSPI_now').text)
"""
$ python Kospi.py
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
$ python Kospi.py
현재 코스피 지수 = 2,060.80
"""
```

- **Web에서 원하는 부분의 경로 복사하기**

  응답.select(selector)
  Web 상의 원하는 부분에
  (마우스 우클릭 > 요소 검사) 실행 시 원하는 부분의 코드를 바로 게시
  찾아진 해당 라인에 (마우스 우클릭 > 복사 > CSS 선택자) 선택
  *naver의 경우 (마우스 우클릭 > copy > copy selector) 선택*

```python
# Naver_Ranking.py
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

"""
$ python Naver_Ranking.py
네이버 1위 검색어.      자사고
네이버 2위 검색어.      천우희
네이버 3위 검색어.      토스머니5만원이벤트
네이버 4위 검색어.      위메프투어 아시아나항공
네이버 5위 검색어.      거제 살인사건
네이버 6위 검색어.      23사단
네이버 7위 검색어.      sbs 앵커
네이버 8위 검색어.      안재홍
네이버 9위 검색어.      제주도 상어
네이버 10위 검색어.     분홍코끼리
1.       자사고
2.       천우희
3.       토스머니5만원이벤트
4.       위메프투어 아시아나항공
5.       거제 살인사건
6.       23사단
7.       sbs 앵커
8.       안재홍
9.       제주도 상어
10.      분홍코끼리
"""
```

