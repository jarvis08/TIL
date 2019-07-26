# Python Standard Libraries

---

## string

- `string.replace('변경 전', '변경 후')` : string의 일부를 원하는 글자로 수정

  ```python
  print('touch example_{}.txt'.format(i))
  print(f'touch example_{i}.txt')
  print('touch example'+ str(i) + '.txt')
  ```
  
- `string.isdigit()` : 문자열이 양의 정수로만 구성되어 있는지 판별

  ```python
  # input()은 모든 입력을 string으로 반화
  num = input()
  if num.isdigit():
      print('숫자입니다.')
  else:
      print('문자입니다.')
  ```

- `string.isalpha()` : 문자열의 내용이 알파벳으로만 구성되어 있는지 판별

- `.strftime()` : 날짜/시간 format에서 날짜/시간 요소만 추출

  `.isoformat()` : %를 사용하지 않으며, 날짜/시간 format을 유지한 채 자료형만 string으로 변환

  ```python
  # datetime.datetime 날짜와 시간이 함께 포함되어 있으므로 date 함수 사용
  from datetime import date, timedelta
  
  yesterday = date.today() - timedelta(days=1)
  print(yesterday)
  yesterday = yesterday.strftime('20%y%m%d')
  print(yesterday)
  """result
  2019-07-16
  20190716"""
  ```

  ```python
  # isoformat은 %를 사용하지 않으며, 날짜/시간 format을 유지한 채 자료형만 string으로 변환
  from datetime import date
  
  yesterday = (date.today() - timedelta(days=1)).isoformat().replace('-', '')
  ```

---

## list

- `list.sort()` : 원본을 오름차순으로 정렬
- `list.sort(reverse=True)` : 원본을 내림차순으로 정렬
- `sorted(list)` : 오름차순한 결과를 반환
- `list.reverse()` : 원본의 순서를 역으로 변환
- `reversed(list)` : 역순으로 변환한 list를 반환
- `list.count(element)` : list 내의 element 개수를 반환

---

## dictionary

- `dic.keys()` : key 값들의 나열을 dictionary의 특수한 자료형으로 나타내어 반환

- `dic.values()` : value 값들의 나열을 dictionary의 특수한 자료형으로 나타내어 반환

- `dic.items()` : (value, key) 값들의 나열을 dictionary의 특수한 자료형으로 나타내어 반환

- `dict.get(key)` : dict[key] 의 결과값을 반환하며, 없을 시 error가 아닌 None을 반환

  ```python
  dictionary = {'first' : {'second' : 3}}
  print(dictionary.get('first').get('second'))
  """result
  3"""
  ```

---

## set

- `set( list )` : list를 집합으로 전환

```python
count = len(set(winner) & set(ur_lotto))
# winner list와 ur_lotto list를 비교할 때
# for문을 이용하는 것 보다 빠른 속도로 같은 요소의 개수를 구함
```

---

## class

- `isinstance(object, class/tuple)` : object가 class/tuple의 instance인지 판별

---

## sort

- `sort()` : list 원본을 오름차순으로 sort
- `sorted()` : 오름차순으로 sort한 결과를 반환

---

## file 'r / w / a'

- `with open('파일명', '파일 조작 유형', encoding='utf-8') as f:``
- `f.readline()` : 한 줄 읽기
- `f.readlines()` : 모든 문장 읽기
- `f.write()` : 한 번 쓰기
- `f.writelines()` : 모두 쓰기
- 파일 조작 3가지
  - `'r'`: read
  - `'w'`: write
  - `'a'`: append

---

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

---

- `filter(function, iterable)`

  function에 의해 처리되는 각각의 요소에 대해 Boolean 값을 반환

  - `True` : 유지
  - `False` : 제거

  ```python
  foo = [2, 9, 27, 3, 4, 5]
  print(list(filter(lambda x: x % 3 == 0, foo)))
  """result
  [9, 27, 3]"""
  ```

---

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

---

## Special Method, Magic Method

- `__method__` 와 같은 형태로 underscore가 두개씩 양 옆에 붙은 method를 의미
- Built-in  Method이며, 기본적으로 python에서 제공

```python
from myPackage.math.formula import pi
dir(pi)
"""result
['__annotations__',
 '__call__',
 '__class__',
 '__closure__',
 '__code__',
 '__defaults__',
 '__delattr__',
 '__dict__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__get__',
 '__getattribute__',
 '__globals__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__kwdefaults__',
 '__le__',
 '__lt__',
 '__module__',
 '__name__',
 '__ne__',
 '__new__',
 '__qualname__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__sizeof__',
 '__str__',
 '__subclasshook__']"""
```

---

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

- `random.sample( [], int )` : [ ] 중 int 개 만큼 비복원 추출

  `random.choice( [ ] )` : [ ] 중 1개를 임의 복원 추출

---

## map( )

```python
# a = input()
# b = a.split(" ")
# num1 = int(b[0])
# num2 = int(b[1])

## 위의 네 줄을 생략할 수 있는 map()
# iterable 한 tuple 형태를 응용

# 8 3 이라고 input을 받아서 사칙연산을 수행하는 예제
num1, num2 = map(int, input().split(" "))
print(num1 + num2)
print(num1 - num2)
print(num1 * num2)
print(num1 / num2)

# iterable 하다면 모두 가능하므로 list로 변형해도 가능
num1, num2 = list(num1, num2 = map(int, input().split(" ")))
```

---

## math

- 함수 종류

  | 함수                | 비고                            |
  | ------------------- | ------------------------------- |
  | math.ceil(x)        | 소수점 올림                     |
  | math.floor(x)       | 소수점 내림                     |
  | math.trunc(x)       | 소수점 버림                     |
  | math.copysign(x, y) | y의 부호를 x에 적용한 값        |
  | math.fabs(x)        | float 절대값 - 복소수 오류 발생 |
  | math.factorial(x)   | 팩토리얼 계산 값                |
  | math.fmod(x, y)     | float 나머지 계산               |
  | math.fsum(iterable) | float 합                        |
  | math.modf(x)        | 소수부 정수부 분리              |

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

---

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
      for column in reader:
          movieCds.append(column['movieCd'])
  ```

---

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

  ---

## timedelta

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

---

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

---

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

---

## time

- 시간 지연

  ```python
  from time import sleep
  # 5초 지연
  sleep(5)
  ```

  