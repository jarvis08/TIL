# SSAFY_Week4_Day1

 **참고자료** : ./50_SSAFY/8ython/notes/05.module.ipynb

---

- 문자열 덧셈 하기

  #### 문자열 조작 및 반복/조건문 활용[¶](http://localhost:8888/notebooks/problems/problem04.ipynb#문자열-조작-및-반복/조건문-활용)

  **문제 풀기 전에 어떻게 풀어야할지 생각부터 해봅시다!**

> 사람은 덧셈을 할때 뒤에서부터 계산하고, 받아올림을 합니다.
>
> 문자열 2개를 받아 덧셈을 하여 숫자를 반환하는 함수 `my_sum(num1, num2)`을 만들어보세요.

```python
def my_sum(num1, num2):
    n1 = list(map(int, num1))
    n2 = list(map(int, num2))
    # 자리수 맞추기
    while True:
        if len(n1) > len(n2):
            n2.insert(0, 0)
        elif len(n1) < len(n2):
            n1.insert(0, 0)
        else:
            break

    # 뒤부터 더해서 올림받기
    up = False
    result = []
    for i in range(1, len(n1)+1):
        tmp = n1[-i] + n2[-i]
        if up:
            tmp += 1
            up = False        
        if tmp >= 10:
            if i == len(n1):
                result.append(f'{tmp}')
                continue
            up = True
            result.append(f'{tmp-10}')
        else:
            result.append(f'{tmp}')
    result.reverse()
    return int(''.join(result))
```

---

## Module, 모듈

- 정의 : 기능별 단위로 분할한 부분

- 모듈화 목적 : 복잡성을 감소시키기 위해 단순화, abstraction(요약)을 하는 수단 중 하나

- 함수 : 가장 기본적인 모듈

- `import` : 모듈을 활용하기 위해서는 반드시 `import` 문을 통해 내장 모듈을 namespace로 불러와야 한다.

  ```python
  # hello.py
  def hi():
      return '안녕하세요'
  
  # module.ipynb
  import hello
  print(hello.hi())
  dir(hello)
  """result
  안녕하세요
  ['__builtins__',
   '__cached__',
   '__doc__',
   '__file__',
   '__loader__',
   '__name__',
   '__package__',
   '__spec__',
   'hi']"""
  ```

  ---

- Package 만들기

  - module의 상/하 구조를 나누어 구조화 할 수 있도록 계층화

  - `__init__.py`

    : 파이썬이 directory를 package로 취급하게 만들기 위해 필요

    : string 처럼 흔히 쓰는 이름의 directory가 의도하지 않게 모듈 검색 경로의 뒤에 등장하는, 올바른 모듈들을 가리는 일을 방지하기 위함

    This prevents directories with a common name, such as string , unintentionally hiding valid modules that occur later on the module search path

  ```python
  /myPackage
      __init__.py
      /math
          __init__.py
          formula.py
      /web
          __init__.py
          url.py
  
  # __init__.py
  pi = 3.145
  
  # formula.py
  def pi():
      return 3.14
  ```

  ---

- `from` 모듈 `import` `attribute`

  `attribute` : 변수(데이터), 함수, 클래스 등을 모두 포함하는 '속성' 값

  ```python
  # formula.py
  from myPackage.math import formula
  
  # math directory의 __init__.py
  from myPackage import math
  
  # __init__.py의 변수 pi
  from myPackage.math import pi
  
  print(math.formula.pi())
  print(math.pi)
  print(pi)
  """result
  3.14
  3.145
  3.145"""
  ```

  ```python
  # 외장 함수 math
  import math
  # 직접 만든 myPackage
  # math까지만 불러오면 외장 함수 math를 덮어씀
  from myPackage.math.formula import pi
  
  print(math.sqrt(4))
  print(pi())
  """result
  2.0
  3.14"""
  ```

  ---

- `myPackage/__init__.py` 안에

  ```python
  from .math import formula
  ```

  기입 시

  ```python
  import myPackage
  ```

  만으로도 `formula.pi()`를 사용 가능

  ```python
  # __init__.py 수정 전
  import myPackage
  dir(myPackage)
  """result
  ['__builtins__',
   '__cached__',
   '__doc__',
   '__file__',
   '__loader__',
   '__name__',
   '__package__',
   '__path__',
   '__spec__']"""
  
  # __init__.py 내부에 from .math import formula 작성 후
  """result
  ['__builtins__',
   '__cached__',
   '__doc__',
   '__file__',
   '__loader__',
   '__name__',
   '__package__',
   '__path__',
   '__spec__',
   'formula',
   'math']"""
  ```

  ---

- `.`을 이용하여 directory를 지정하지만, directory 지정이 `.` 사용의 목표가 아니다.

- `.`은 module의 계층을 구분하는 의미로써, 물리적 directory 구조를 생각하면 오류 발생

  - 따라서 `__init__.py` 내부에 하부 계층의 module을 `import` 해주지 않는 이상,

    `import myPackage` 만으로 하부 계층의 module(ex.`formula.py`)을 사용 불가

  ---

- `__all__=['모듈명', '모듈명']`

  `__init__.py`에 package의 사용할 모듈 모두 지정해 두기

  ```python
  # __init__.py에 추가하기
  __all__ = ['math','web']
  
  # 다른 곳에서 *를 사용하여 불러오기
  from myPackage import *
  """result
  ['__all__',
   '__builtins__',
   '__cached__',
   '__doc__',
   '__file__',
   '__loader__',
   '__name__',
   '__package__',
   '__path__',
   '__spec__',
   'math',
   'web']"""
  ```

  ---

- python version 3.3 이후로는 `__init__.py`를 작성하지 않더라도 실행 가능

- 현재는 하위 호환성을 위해 계속 작성

  ---

- `from` 모듈 `import` attribute `as` 별칭

  : 내가 지정하는 이름으로 `import`

  ```python
  from bs4 import BeautifulSoup as bs
  ```

---

## Python Standard Library

- library path

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

- magic method

  내가 제작한 modul일 지라도, python이 알아서 추가한 기능

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

- `math`

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

    ```python
    ​```
    sin, cos, tan
    asin, acos, atan, 
    sinh, cosh, tanh,
    ashinh, acosh, atanh
    ​```
    ```

- random

  - 난수 생성

    ```python
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

- datetime

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

    ```
    datetime.date(year, month, day, hour, minute, second, microsecond)
    ```

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

  - `timedelta`

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

    

---

## 예제

- 숫자 패턴

  ```python
  # 다음과 같은 함수 만들기
  print(pattern(9)):
  1
  333
  55555
  7777777
  999999999
  print(pattern(6))
  1
  333
  55555
  ```

  ```python
  # 일반적인 방법
  def pattern(n):
      result = ''
      for turn in range(1, n+1, 2):
          result += f'{str(turn) * turn}\n'
      return result
  print(pattern(9))
  print(pattern(6))
  
  # lambda
  pattern = lambda n : "\n".join([str(i)*i for i in range(n+1) if i %2])
  print(pattern(9))
  print(pattern(6))
  ```

---

## HASH

- `sha256` : Web의 보안 Hash 생성 방법
  - 2**256의 경우의 수로 랜덤 생성
- git log를 통해 hash를 확인 가능(sha256 급의 보안은 아님!)
  - hash로 인해 git 내용의 철자 하나만 바뀌어도 즉각적인 변화 포착

---

# Errors and Exceptions

 **참고자료** : ./50_SSAFY/8ython/notes/06.erros.ipynb

- 문법 에러, Syntax Error

  : 가장 많이 만날 수 있는 에러로 발생한 `파일 이름`과 `줄`, `^`을 통해 파이썬이 읽어 들일 때(parser)의 문제 발생 위치를 표현한다.

- `SyntaxError`

  ```python
  # 문법 에러, 무조건 내 잘못
  # : 안닫음
  if True
      print('참')
  """result
  SyntaxError: invalid syntax"""
  ```

- `EOL`, `EOF`

  ```python
  # End of Line, End of File
  # ' ", ) ] 안닫음
  print('hello)
  """result
  SyntaxError: EOL while scanning string literal"""
  ```

- `ZeroDivisionError`, `NameError`, `TypeError`

- argument 누락/초과

  ```python
  # 몇개 뽑을 것인지 선언 안함
  random.sample(range(45))
  ```

- `ValueError` : 자료형은 올바르지만, 값이 틀린 경우

  ```python
  int('3.5')
  ```

- `IndexError`, `KeyError`, `ModuleNotFoundError`, `ImportError`

- `KeyboardInterrupt`

  ```python
  while True:
      continue
  # 무한 반복 중에 'ctrl + C'와 같은 강제 중단 명령이 있을 경우
  ```