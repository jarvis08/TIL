# Python Standard Libraries

## os

- `os.system('CLI commands')`: CLI에서 사용하는 명령어를 그대로 사용

  ```python
  os.system('touch example.txt')
  # example.txt 파일 생성
  os.system('rm example.txt')
  # example.txt 파일 제거
  ```

- `os.listdir()`: 현재 디렉토리 내부의 모든 파일, 디렉토리를 리스트에 저장

- `os.rename(현재 파일명, 바꿀 파일명)`: 파일명 변경

- `os.join(path, file_name)` : path 결합

- 상위 directory로 거슬러 올라가기

  ```python
  import sys
  # 한 단계 위의 directory file을 사용하고 싶을 때
  sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
  ```

| code                                                         | info                                                         | output                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------- |
| `os.getcwd()`                                                | 현재 작업 폴더                                               | "C:/Temp"                                |
| `os.chdir()`                                                 | 작업 디렉토리를 현 위치에서 해당 위치로 옮김                 |                                          |
| `os.path.abspath("./Scripts")`                               | 특정 경로에 대해 절대 경로 얻기                              | "C:/Python35/Scripts"                    |
| `os.path.dirname("C:/Python35/Scripts/pip.exe")`             | 경로 중 디렉토리명 얻기                                      | "C:/Python35/Scripts"                    |
| `if os.path.isfile("C:/Python35/Scripts/pip.exe"):    print(os.path.basename("C:/Python35/Scripts/pip.exe"))` | 경로 중 파일명만 얻기                                        | "pip.exe"                                |
| `dir, file = os.path.split("C:/Python35/Scripts/pip.exe")`   | 경로 중 디렉토리명과 파일명을 나누어 얻기                    |                                          |
| `"C:\Python35\Scripts\pip.exe".split(os.path.sep)`           | 파일 각 경로를 나눠 리스트로 리턴하기 os.path.sep은 OS별 경로 분리자 | ['C:', 'Python35', 'Scripts', 'pip.exe'] |
| `os.path.join('C:\Tmp', 'a', 'b')`                           | 경로를 병합하여 새 경로 생성                                 | "C:\Tmp\a\b"                             |
| `os.listdir("C:\Python35")`                                  | 디렉토리 안의 파일/서브디렉토리 리스트                       |                                          |
| `os.path.exists("C:\Python35")`                              | 파일 혹은 디렉토리 경로가 존재하는지 체크하기                |                                          |
| `os.path.isdir("C:\Python35")`                               | 디렉토리 경로가 존재하는지 체크하기                          |                                          |
| `os.path.isfile("C:\Python35\python.exe")`                   | 파일 경로가 존재하는지 체크하기                              |                                          |
| `os.path.getsize("C:\Python35\python.exe")`                  | 파일의 크기                                                  |                                          |

- 예시

  - `os.getcwd()`

    current working directory

  - `os.path.join(a, b, c)`

    `a/b/c` 형태로 `a` 경로에 `b`와 `c`의 경로를 추가

    ```python
    import os
    current = os.getcwd()
    templates_path = os.path.join(current, 'Directory_name')
    ```

## sys

```python
# 현재 directory를 시작으로,
# 이후 system path를 뒤져가며 해당 모듈을 search
import sys
sys.path
"""result
['C:\\Users\\student\\TIL\\50_SSAFY\\8ython\\notes',
 'c:\\users\\student\\appdata\\local\\programs\\python\\python37-32\\python37.zip',
 'c:\\users\\student\\appdata\\local\\programs\\python\\python37-32\\DLLs',
 'c:\\users\\student\\appdata\\local\\programs\\python\\python37-32\\lib',
 'c:\\users\\student\\appdata\\local\\programs\\python\\python37-32',
 '',
 'C:\\Users\\student\\AppData\\Roaming\\Python\\Python37\\site-packages',
 'c:\\users\\student\\appdata\\local\\programs\\python\\python37-32\\lib\\site-packages',
 'c:\\users\\student\\appdata\\local\\programs\\python\\python37-32\\lib\\site-packages\\IPython\\extensions',
 'C:\\Users\\student\\.ipython']"""
```

<br><br>

## random

- 난수 생성

  ```python
  import random
  # 0 ~ 1 사이 임의의 실수
  random.random()
  
  # 1이상 100이하(미만x)의 정수 생성
  random.randint(1, 100)
  ```

- 시드 설정

  ```python
  # 원하는 seed를 부여하면 항상 동일한 random 값 반환
  random.seed(1111)
  """result
  0.21760077176688164"""
  ```

- `random.shuffle(iterable)` : iterable을 섞음

  ```python
  names = [1, 2, 3, 4]
  random.shuffle(names)
  print(names)
  ```

- Sampling

  - `random.sample( [], int )`
  
    [ ] 중 int 개 만큼 비복원 추출
    
  - `random.choice( [ ] )`
  
    [ ] 중 1개를 임의 복원 추출

<br><br>

## math

- 함수 종류

  | 함수                | 비고                                                     |
  | ------------------- | -------------------------------------------------------- |
  | math.ceil(x)        | 소수점 올림                                              |
  | math.floor(x)       | 소수점 내림                                              |
  | math.trunc(x)       | 소수점 버림                                              |
  | math.copysign(x, y) | y의 부호를 x에 적용한 값                                 |
  | math.fabs(x)        | float 절대값 - 복소수 오류 발생                          |
  | math.factorial(x)   | 팩토리얼 계산 값                                         |
  | math.fmod(x, y)     | float 나머지 계산                                        |
  | math.fsum(iterable) | float 합                                                 |
  | math.modf(x)        | 소수부 정수부 분리                                       |
  | math.isclose(x, y)  | x, y가 같거나, 같다고 말 할 수 있을 정도로 차이가 없는가 |

  ```python
  # 내림과 버림은 음수에서 처리가 다르다.
  # 내림은 정수부와 소수부가 존재하며, 소수부는 양수이다.
  print(math.floor(-pi))
  print(math.trunc(-pi))
  """result
  -4
  -3"""
  ```

- 로그, 지수 연산

  | 함수                | 비고                  |
  | ------------------- | --------------------- |
  | math.pow(x,y)       | x의 y승 결과          |
  | math.sqrt(x)        | x의 제곱근의 결과     |
  | math.exp(x)         | e^x 결과              |
  | math.log(x[, base]) | 밑을 base로 하는 logx |

- 삼각함수

  ~~~python
  ```
  sin, cos, tan
  asin, acos, atan, 
  sinh, cosh, tanh,
  ashinh, acosh, atanh
  ```
  ~~~

<br><br>

## CSV

- CSV = Comma Seperated Values

  - `csv.writer(파일명)` : '파일명'을 조작하는 writer를 원하는 이름으로 생성

    `writer.writerow(iterable_item)` : iterable_item을 전에 생성한 writer를 사용하여 한줄씩 끊어서 작성(writerow)

  - `csv.DictWriter(파일명, fieldnames=필드명_list)`

    이 함수를 선언시, `writer.writerow(iterable)`을 통해 dictionary를 그대로 넣어도 csv 형태 작성

    `writer.writeheader()` : 해당 코드를 넣어두면 fieldnames까지 첫 줄에 작성

  ```python
  # csv 파일 만들기
  lunch = {
      '진가와' : '01011112222',
      '대우식당' : '01054813518',
      '바스버거' : '01088465846'
  }
  # 1. lunch.csv 데이터 저장
  with open('lunch.csv', 'w', encoding='UTF-8') as f:
      for k, v in lunch.items():
          f.write(f'{k},{v}\n')
  
  # 2. ',' join을 사용하여 string 만들기
  with open('lunch.csv', 'w', encoding='UTF-8') as f:
      for item in lunch.items():
          f.write(','.join(item))
  
  # 3. csv 라이브러리 사용
  # writer와 reader가 따로 존재
  import csv
  with open('lunch.csv', 'w', encoding='utf-8', newline='') as f:
      csv_writer = csv.writer(f)
      for item in lunch.items():
          csv_writer.writerow(item)
          
  # 4. csv.DictWriter()
  # field name을 설정 가능
  with open('student.csv', 'w', encoding='utf-8', newline='') as f:
      fieldnames = ['name', 'major']
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerow({'name':'john', 'major':'cs'})
      writer.writerow({'name':'dongbin', 'major':'ie'})
  ```
  
  ```python
  # csv Reader
  with open('boxoffice.csv', 'r', encoding='utf-8') as f:
      # reader instance 생성
      reader = csv.DictReader(f)
      # reader를 이용하여 movieCd 열의 data를 parse
      for row in reader:
          movieCds.append(row['movieCd'])
  ```

<br><br>

## datetime

- `datetime.date(year, month, day, hour, minute, second, microsecond)`

- method

  | 속성/메소드 | 내용                 |
  | ----------- | -------------------- |
  | .year       | 년                   |
  | .month      | 월                   |
  | .day        | 일                   |
  | .hour       | 시                   |
  | .minute     | 분                   |
  | .second     | 초                   |
  | .weekday()  | 월요일을 0부터 6까지 |

  ```python
  # 1970년 1월 1일부터 1초씩 증가
  # 오늘
  print(datetime.today())
  """result
  2019-07-22 16:20:41.031672"""
  # 2019년 7월 22일 16시 18분 54초
  
  datetime.today()
  """result
  datetime.datetime(2019, 7, 22, 16, 20, 59, 338339)"""
  
  # UTC기준시
  print(datetime.utcnow())
  """result
  2019-07-22 07:21:34.658591"""
  ```

- 시간 형식 지정

  | 형식 지시자(directive) | 의미                   |
  | ---------------------- | ---------------------- |
  | %y                     | 연도표기(00~99)        |
  | %Y                     | 연도표기(전체)         |
  | %b                     | 월 이름(축약)          |
  | %B                     | 월 이름(전체)          |
  | %m                     | 월 숫자(01~12)         |
  | %d                     | 일(01~31)              |
  | %H                     | 24시간 기준(00~23)     |
  | %I                     | 12시간 기준(01~12)     |
  | %M                     | 분(00~59)              |
  | %S                     | 초(00~61)              |
  | %p                     | 오전/오후              |
  | %a                     | 요일(축약)             |
  | %A                     | 요일(전체)             |
  | %w                     | 요일(숫자 : 일요일(0)) |
  | %j                     | 1월 1일부터 누적 날짜  |

  ```python
  # strf : time을 string format으로
  now = datetime.now()
  now.strftime('%Y %m %d %A')
  """relsult
  '2019 07 22 Monday'"""
  ```

- 특정한 날짜 만들기

  ```python
  # 크리스마스 만들기
  christmas = datetime(2018, 12, 25)
  print(christmas)
  """result
  2018-12-25 00:00:00"""
  print(christmas.strftime('%d'))
  """result
  '25'"""
  ```

<br>

### timedelta

```python
from datetime import timedelta
ago = timedelta(days=3)
print(ago)
"""result
3 days, 0:00:00"""
now = datetime.now()
print(now - ago)
"""result
2019-07-19 16:24:53.255855"""
```

```python
# 예시 코드
# 어제 날짜 구하기
from datetime import date, timedelta
def my_url(key, targetDt=None):
    if targetDt == None:
        # strftime에서는 % 사용
        # yesterday = date.today() - timedelta(days=1)
        # targetDt = yesterday.strftime('20%y%m%d')
        targetDt = (date.today() - timedelta(days=1)).isoformat().replace('-', '')
    api['targetDt'] = targetDt
    base_url = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json?'
    request_url = f'{base_url}key={key}&targetDt={targetDt}'
    return request_url
```

<br><br>

## collections

- 내장 함수이며, 알고리즘 속도 개선을 위해 자주 사용

- dictionary와 유사하지만, 다른 type

  ```python
  import collections
  blood_types = ['A', 'B', 'A', 'O', 'AB', 'AB', 'O', 'A', 'B', 'O', 'B', 'AB']
  print(collections.Counter(blood_types))
  """python
  Counter({'A': 3, 'B': 3, 'O': 3, 'AB': 3})"""
  ```

- dictionary 정렬하기

  ```python
  import collections
  order_dict = collections.OrderedDict(dict)
  ```

- deque, double-ended queue

  `appendleft()`

  `extendleft()`

  `popleft()`

  `rotate(int_n)` : 정수(음수 가능) 값 만큼 요소를 회전(순서 밀어내기)

  ```python
  import collections
  
  deq = collections.deque(['a', 'b', 'c'])
  deq.appendleft('d')
  print(deq)
  '''result
  deque(['d', 'a', 'b', 'c'])
  '''
  
  deq1 = collections.deque(['a', 'b', 'c', 'd', 'e'])
  deq1.rotate(-2)
  print('deq1 >>', ' '.join(deq1))
  '''result
  deq1 >> c d e a b
  '''
  ```

<br><br>

## functools

- `functools.reduce(function, iterable, initializer=None)`

  function을 통해 iterable을 하나의 값으로 줄인다.

  initializer가 주어지면 첫 번째 인자로 추가

  ```python
  from functools import reduce
  reduce(function, [1, 2, 3, 4, 5])
  1. reduce(function, [function(1, 2), 3, 4, 5])
  2. reduce(function, [function(function(1,2), 3), 4, 5]
  ...
  ```

<br><br>

## time

- 시간 지연

  ```python
  from time import sleep
  # 5초 지연
  sleep(5)
  ```

<br><br>

## copy

### Deep Copy

```python
import copy
matrix_1 = [[1, 2], [3, 4]]
matrix_2 = copy.deepcopy(matrix_1)
print(matrix_2[0][0])
```

<br><br>

## heapq

`heapq` 모듈은 list를 최소힙으로 사용할 수 있게 한다. heapq의 대표적인 메서드는 다음과 같다.

- `heapify(list)`: list를 힙으로 변환
- `heappush(list, element)`: list이 힙의 형태를 유지하도록, element를 삽입
- `heappop(list)`: list의 `0` index, 최소값을 `pop` 하며, 나머지 요소들을 다시 heap 형태로 유지

```python
import heapq

heap = []
a = [3, 1, 8, 5, 12, 30, 7]
for i in range(len(a)):
    heapq.heappush(heap, a[i])
print(heap)

heapq.heapify(a)
print(a)

for _ in range(len(heap)):
    print(heapq.heappop(heap))

"""result
[1, 3, 7, 5, 12, 30, 8]
[1, 3, 7, 5, 12, 30, 8]
1
3
5
7
8
12
30"""
```

