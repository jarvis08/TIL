# SSAFY_Day_1

## Markdown Language

- `- + Enter`: black point

- `(```python + Enter)`: 블록 코드 강조 (back tick)

  ```python
  (Ctrl + Enter) : 빠져나가기
  ```

- `--- + Enter`: 줄긋기

- `Ctrl + T`: 표 만들기
- `back tick + 내용 + back tick`: 코드블록에 내용 넣기

---

## Turing Completeness 

- 저장 :: 어디에 무엇을 어떻게 넣는가?
  1. 숫자
  2. 글자
  3. 참/거짓
- 조건
- 반복

---

## Python은 어떻게 저장하는가?

- 변수 (variable) 

​	:: 박스 1개

- 리스트 (list)

​	:: 박스 여러개

​		list = ["강남구", ...]

- 딕셔너리 (dictionary)

​	:: 견출지 붙인 박스들의 묶음

​		dic = {"강남구" : 50, ...}

```python
# random.choice()
import random
menu = ['피자', '장어구이', '삼겹살', '고등어구이']
choice = random.choice(menu)
print(choice)
```

```python
# 미세먼지 알리미
import requests
from bs4 import BeautifulSoup
url = f'http://openapi.airkorea.or.kr/openapi/services/rest/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty?serviceKey={key}&numOfRows=10&pageSize=10&pageNo=3&startPage=3&sidoName=%EC%84%9C%EC%9A%B8&ver=1.6'
response = requests.get(url).text
soup = BeautifulSoup(response,'xml')
gangnam = soup('item')[7]
location = gangnam.stationName.text
time = gangnam.dataTime.text
dust = int(gangnam.pm10Value.text)
# pm10Value :: 미세먼지농도

# print('{0} 기준 {1}의 미세먼지 농도는 {2}입니다.'.format(time,location,dust))

# dust 변수에 들어 있는 내용을 출력해보세요.
if dust > 150:
  ko = "매우나쁨"
elif dust > 80:
  ko = "나쁨"
elif dust > 30:
  ko = "보통"
else:
  ko = "좋음"
print("{}, {}".format(dust, ko))

'''
14, 좋음
'''
```

```python
random.choice(list) :: 임의 복원 추출
random.sample(list, num) :: num만큼 비복원추출
```

```python
# 1. random 외장 함수 가져오기
import random
# 2. 1~45까지 숫자 numbers에 저장하기
num = range(1,46)
print(type(num))
# num = []
# for i in range(45):
#   num[i] = i+1
# """
# 3. numbers에서 6개 뽑기
num = random.sample(num, 6)
# 4. 출력하기
print(sorted(num))
# 5. 한줄로 줄이기
print(sorted(random.sample(range(1,46), 6)))

'''
<class 'range'>
[6, 7, 10, 14, 24, 41]
[2, 7, 8, 19, 22, 31]
'''
```

---

## CLI(Command Line Interface)

: 유닉스 shell(Linux, Git Bash, ...)

: CMD, Powershell

`ls` : list 목록

`cd`: 지정 위치로 이동

`pwd`: 현위치 (point working directory)

`mkdir`: 폴더 생성

`code .`: vs code 현위치에서 실행

`touch 문서.txt`: 문서를 생성

`rm 문서.txt`: 문서 삭제

- chocolatey 설치 추천

  : pip처럼 여러 패키지를 Win cmd에서 설치 및 삭제 가능
  
- VS Code 환경설정

  1. Git Bash 상에서 code . 을 사용하여 code 실행
  2. `Ctrl + Shift + P`: shell을 검색하여 default shell을 git bash로 설정
  3. Terminal을 이용하여 실행

---

## URL tip

daum을 통해 아이유 를 검색할 때,

https://search.daum.net/search?w=tot&DA=YZR&t__nil_searchbox=btn&sug=&sugo=&q=%EC%95%84%EC%9D%B4%EC%9C%A0](https://search.daum.net/search?w=tot&DA=YZR&t__nil_searchbox=btn&sug=&sugo=&q=수지

중에서(위 url은 daum 페이지에서 아이유를 검색했을 때, 주소창에 적힌 코드)

https://search.daum.net/search?q=수지

부분만 있어도 검색이 가능하다! 나머지는 부가 옵션

```python
import webbrowser

url = "https://search.daum.net/search?q="
keywords = ["수지", "한지민"]
for keyword in keywords:
    webbrowser.open(url + keyword)
```

---

## Git

- 생활코딩 git 강의 좋다

```shell
cd (master directory)
git init
git add .
git commit -m "1"
"""
Run

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"
"""
git config --global user.email "cdb921226@gmail.com"
git config --global user.name "jarvis08"
git remote add origin https://github.com/jarvis08/SSAFY.git
git push -u origin master
```

