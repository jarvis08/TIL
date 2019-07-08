# SSAFY_Day1

## Markdown Language

- (- + Enter) :: black point

------

- (```python + Enter) :: 블록 코드 강조 (back tick)

  ```python
  (Ctrl + Enter) :: 빠져나가기
  ```

---

- ((---) + Enter) :: 줄긋기

---

- (Ctrl + T) :: 표 만들기



## Day 1

- Turing Completeness 튜링 완전
  - 저장 :: 어디에 무엇을 어떻게 넣는가?
    1. 숫자
    2. 글자
    3. 참/거짓
  - 조건
  - 반복

---

- 어떻게 저장하는가?

  1) 변수 (variable) 

  ​	:: 박스 1개

  2) 리스트 (list)

  ​	:: 박스 여러개

  ​		list = ["강남구", ...]

  3) 딕셔너리 (dictionary)

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
```

- 공공데이터 포털 :: data.go.kr