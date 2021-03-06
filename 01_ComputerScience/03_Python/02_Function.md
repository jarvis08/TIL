# Function

 **참고자료** : ./50_SSAFY/8ython/notes/03.funtion.jpynb

## 함수, Function

- 복잡한 세상을 컴퓨터에게 단순하게 이해시키도록 하는 것이 사람이 할 일

  세상 문제들의 **`Complexity`**를 낮추기 위해 **`Abstraction`**(추상화 X, **요약** O)을 이용하는 것이 핵심

- **`함수, Function`**은 **`Abstraction`**의 수단 중 하나

- **함수의 선언과 호출**

  ```python
  def func(parameter1, parameter2):
      code line1
      code line2
      return value
  ```

  - 함수 선언은 `def`로 시작하여 `:`으로 끝나고, 다음은 `4spaces 들여쓰기`

  - 함수는 `매개변수(parameter)`전달이 가능

  - 함수는 동작후에 `return`을 통해 결과값을 전달 가능 

    (`return` 값이 없으면 None을 반환, e.g., `print()`)

  - `func(val1, val2)` : 함수는 호출

- **내장 함수**

  ```python
  # 내장 함수 확인
  dir(__builtins__)
  ```

- 함수의 **`return`**

  - 함수는 반환되는 값이 있으며, 이는 어떠한 종류의 객체여도 무관
  - **단, 오직 한 개의 객체만 반환** 가능
  - multiple values를 반환한 것 같아도, **`tuple`**형태로 하나의 객체로 반환

<br>

<br>

## 인자(Parameter) & 인수(Argument)

- 함수의 **인수**

  : 함수는 `인수, argument` 전달이 가능

  - 위치 인수

    : 기본적으로 함수는 인자와 인수를 위치로 판단

    ```python
    # 인자 및 인수의 순서를 바꾸면 다른 값을 return
    def cylinder(r, h):
        return r**2 * h * 3.14
    ```

<br>

### **기본 값, Default Argument Values**

: 함수가 호출될 때, 인자를 지정하지 않아도 기본 값을 설정 가능

**단, 기본값을 갖는 인자의 위치는 가장 마지막**

```python
def ex_func(param_1, default_param=default_value):
    return param_1 + p1
a = ex_func(3)
print(a)
"""result
3 + default_value"""
```

<br>

### **키워드 인자, Keyword Arguments**

`**kwargs` : **keyword argument

: 키워드 인자는 직접적으로 **변수의 이름**으로 **특정 인자**를 전달

```python
def greeting(age, name='john'):
    print(f'{name}은 {age}살입니다.')
    pass
greeting(24, name='철수')
```

```python
# print(*object, sep='', end='\n', file=sys.stdout, flush=False)
print('첫번째 문장')
print('두번째 문장', end='_')
print('세번째 문장', '마지막 문장', sep="/", end="끝!")
"""result
첫번째 문장
두번째 문장_세번째 문장/마지막 문장끝!"""
```

```python
  # 인자 전달
  def ssafy(name, location='서울')
  	print(f'{name}의 지역은 {location}입니다.')
  
  # 가능
  ssafy(name='철수', location='대전')
  ssafy(location='대전', name='철수')
  
  # 불가능
  ## keyword argument는 positional argument 보다 뒤쪽에 위치해야함
  ssafy(name='철수', '대전')
```

<br>

### **가변 인자 리스트**

`*args` : *arguments

`print()`처럼 **정해지지 않은 임의의 개수의 인자**를 받기 위해서는 가변인자를 활용

가변인자는 **`tuple` 형태**로 처리가 되며, `*`(asterisk)로 표현

`def func(*args):`

```python
# 가변 인자
def my_max(*nums):
    return max(*nums)

# 응용
print([*range(10)][1::2])
"""result
[1, 3, 5, 7, 9]"""
```

<br>

### **정의되지 않은 인자**들 처리하기

정의되지 않은 인자들은 **`dict` 형태**로 처리되며, __`**`로 표현__

**주로 `kwagrs`라는 이름**을 사용하며, `**kwargs`를 통해 인자를 받아 처리

- `def func(**kwargs):`
  - `func(키워드 = 특정 값)` 형태로 함수 호출
  - `{'키워드' : 특정 값}` 딕셔너리 형태로 함수로 전달

```python
def fake_dict(**kwargs):
    print(kwargs)
fake_dict(한국어='안녕', 영어='hi', 독일어='Guten Tag')

"""result
{'한국어': '안녕', '영어': 'hi', '독일어': 'Guten Tag'}"""
```

```python
# flask에서의 예시
# 항상 다음과 같이 return 지정 할 때,
@app.route('/')
def main():
    return render_template('index.html', name=name)

# 아래와 같은 함수가 있기 때문에 가능했던 것
def render_template(template_name, **kwargs)
	kwargs['name'] = name
```

```python
# 정의하지 않고 인자 전달 받기
def user(**kwargs):
    if kwargs['password'] == kwargs['password_confirmation']:
        print('회원가입이 완료됐습니다.')
    else:
        print('비밀번호가 일치하지 않습니다')
        
my_account = {
    'username': '홍길동',
    'password': '1q2w3e4r',
    'password_confirmation': '1q2w3e4r'
}
user(**my_account)

"""result
홍길동님, 회원가입이 완료되었습니다."""
```

<br>

### **dictionary를 인자로 넘기기, unpacking arguments list**

__`**dict`__를 통해 함수에 인자 전달 가능

```python
def user(username, password, password_confirmation):
    if password == password_confirmation:
        print(f'{username}님, 회원가입이 완료되었습니다.')
    else:
        print('비밀번호와 비밀번호 확인이 일치하지 않습니다.')

my_account = {
    'username': '홍길동',
    'password': '1q2w3e4r',
    'password_confirmation': '1q2w3e4r'
}
# dictionary를 전달한다고 **를 통해 인지
user(**my_account)

"""result
홍길동님, 회원가입이 완료되었습니다."""
```

<br>

<br>

## Namespace and Scope, 이름공간 및 스코프

- 파이썬에서 사용되는 이름들은 이름공간(Namespace)에 저장

- **LEGB Rule** 적용

  변수에서 값을 찾을 때 **아래와 같은 순서**대로 이름을 탐색

  - `L`ocal Scope: 정의된 함수

  - `E`nclosed Scope: 상위 함수

  - `G`lobal Scope: 함수 밖의 변수 혹은 import된 모듈

  - `B`uilt-in Scope: 파이썬안에 내장되어 있는 함수 또는 속성

    ```python
    # Built-in Scope
    # Global Scope
    def func():
    	# Local Scope
        pass
    
    for i in range(5):
        # Enclosed Scope
        for j in range(5):
            # Local Scope
    ```

  - Namescope 수명주기

    - `Built-in Scope `: 파이썬이 실행된 이후부터 끝까지
    - `Global Scope` : 모듈이 호출된 시점 이후 혹은 이름 선언된 이후부터 끝까지
    - `Local/Enclosed Scope` : 함수가 실행된 시점 이후부터 리턴할때 까지

- 함수는 1급 객체

  함수명이 이전의 변수 선언을 덮어 쓰므로 이름의 차이 필요

- built-in frame 또한 변수처럼 저장 가능한 객체

  ```python
  # built-in frame인 str의 역할을 해줄 str_func 생성
  print(type(str))
  """result
  <class 'type'>"""
  
  str_func = str
  str_func(123)
  str = 'hello'
  
  print(type(str_func))
  print(str)
  """result
  <class 'type'>
  hello"""
  
  # 다시 되돌리기 가능
  str = str_func
  ```

- Local Scope에서 Global Scope를 수정 가능한 경우

  무슨 짓을 해도 local에서 global을 수정할 수 없지만, 한 가지 방법 존재

  ```python
  # global 선언 후 수정
  global_num = 10
  def funct():
      global global_num
      global_num = 5
  funct()
  print(global_num)
  ```

<br>

<br>

## Recursive Function, 재귀함수

- Computational method로써 중요

  작은 문제들을 반복해서 해결하여 큰 문제를 수월하게 해결할 수 있도록 사용

- 하지만 시간 복잡도 증가, 혹은 Stack Overflow(메모리 스택 초과)로 인해 속도 저하

- Algorithm Test에서는 최적해가 존재하지 않는 경우인 Dynamic Programming 문제를 풀 때에만 사용 권장

- 메모리 공간에 쌓이는 모습을 보고 싶다면 `Python Tutor` 이용

- 재귀 함수는 기본적으로 같은 문제이지만 **점점 범위가 줄어드는 문제**를 풀게 된다.

- 재귀함수 작성시 반드시, **`base case` 필요**

- **`base case`** : 점점 범위가 줄어들어 반복되지 않는 **최종적으로 도달하는 곳**

  `Factorial` 계산에서의 `base case`는 n이 1일때, 함수가 아닌 정수 반환하는 것이다.

- Tower of Hanoi, 하노이 탑 참조

  <https://ko.khanacademy.org/computing/computer-science/algorithms/towers-of-hanoi/a/towers-of-hanoi>

  ```python
  # Factorial 구현
  def fact(n):
      if n == 1:
          return 1
      return n * fact(n-1)
  print(fact(5))
  """result
  120"""
  
  # Fibonacci, Recursive
  def fib(n):
      # base case
      if n == 0 or n==1:
          return 1
      else:
          return fib(n-1) + fib(n-2)
  print(fib(4))
  """result
  5"""
  
  # Fibonacci, for문
  def fib_loop(n):
      a, b = 1, 1
      for i in range(n-1):
          a, b = b, a + b
      return b
  
  # 하노이 탑
  ```

<br>

<br>

## 예제 1. URL 편하게 만들기

### 영진위에서 제공하는 일별 박스오피스 API 서비스 이용하기

```
기본 요청 URL : http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json?
```

- key : 발급받은 키값(abc)
- targetDt : yyyymmdd
- itemPerPage : 1 ~ 10 **기본 10**

<br>

### URL 검증하기

만들어진 요청을 보내기전에 URL을 검증

검증 로직 구현 및 문자열 반환

```python
from datetime import date, timedelta
# datetime.datetime 날짜와 시간이 함께 포함되어 있으므로 date 함수 사용
# timedelta는 시간차이를 계산하기 위해 사용(years, hours, days)

# targetDt=None 이라는 default 인자를 주어 인자 전달이 안되어도 에러가 안나도록 함
# Default를 yesterday로 설정해버리면 줄은 짧아지지만, 코드 보기가 더러움
def my_url(key, targetDt=None):
    if targetDt == None:
        # strftime은 % 사용해야 하기 때문에 꺼려지는 경우
        # yesterday = date.today() - timedelta(days=1)
        # targetDt = yesterday.strftime('20%y%m%d')
        targetDt = (date.today() - timedelta(days=1)).isoformat().replace('-', '')
    api['targetDt'] = targetDt
    base_url = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json?'
    request_url = f'{base_url}key={key}&targetDt={targetDt}'
    return request_url

api = {
    'key' : '430156241533f1d058c603178cc3ca0e'
}
targetDt = input('원하는 날짜')
if targetDt:
    api['targetDt'] = targetDt
my_url(**api)
```

<br>

<br>

## 예제 2. 이진법으로 제곱근의 근사값 구하기

- Bisection search

```python
import math
def my_sqrt(m):
    x, y = 1, n
    result = 1
    while abs(result**2 - n) > 0.0000001:
        result = (x+y) / 2
        if result ** 2 < n:
            x = result
        else:
            y = result
    return result

print(like_sqrt(2))
print('오차 = ', math.sqrt(2) - like_sqrt(2)[0])

"""result
(1.414213562373095, 1.4142135623730958)
오차 = 2.220446049250313e-16"""
```

<br>

<br>

## Iterable에 사용가능한 함수

### `map()`, `filter()`, `zip()` 이외의 함수들

- `all()` : all은 인자로 받는 `iterable`(`range`, `list` 등)의 **모든 요소가 참**이거나 **비어있으면** `True`를 반환

  ```python
  print(all([1, 2, 5, '6']))
  print(all([[], 2, 5, '6']))
  """result
  True
  False"""
  
  # all() 만들어보기
  def my_all(x):
      for element in x:
          # 하나라도 False면 의미X
          # [[], 1] 의 경우 빈 list일지라도 존재하기 때문에 True로 처리
          if not element:
              return False
      return True
  ```

- `any()` : any는 인자로 받는 `iterable`(`range` 혹은  `list`)의 요소 중 하나라도 참이면 True를 반환하고, 비어있으면 False를 반환

  ```python
  print(any([1, 2, 5, '6']))
  print(any([[], 2, 5, '6']))
  print(any([0]))
  """result
  True
  True
  False"""
  
  # any() 만들어보기
  def my_any(x):
      for element in x:
          if element:
              return True
      return False
  ```

- 소수 찾기

  ```python
  numbers = [26, 39, 51, 53, 57, 79, 85]
  
  def sosu(l):
      for num in l:
          for i in range(2, num):
              if num % i == 0:
                  print(f'{num}은(는) 소수가 아닙니다. {i}는 {num}의 인수입니다.')
                  break
          else:
              print(f'{num}은(는) 소수입니다.')
              
  sosu(numbers)
  """python
  26은(는) 소수가 아닙니다. 2는 26의 인수입니다.
  39은(는) 소수가 아닙니다. 3는 39의 인수입니다.
  51은(는) 소수가 아닙니다. 3는 51의 인수입니다.
  53은(는) 소수입니다.
  57은(는) 소수가 아닙니다. 3는 57의 인수입니다.
  79은(는) 소수입니다.
  85은(는) 소수가 아닙니다. 5는 85의 인수입니다."""
  ```

- 최대공약수, 최소공배수 구하기

  ```python
  # 유클리드 호제법
  # GCD = Greatest Common Divisor
  # LCM = Least Common Multiple
  
  # GCD/LCM 구하기
  def gcdlcm(a, b):
      # max, min을 할 필요 없음
      # 어차피 작은 수를 큰 수로 나누면 나머지는 작은수
      # m, n = max(a, b), min(a, b)
      m, n = a, b
      while n > 0:
          m, n = n, m % n
      return [m, int(a*b / m)]
  print(gcdlcm(3, 12))
  print(gcdlcm(1071, 1029))
  
  
  # 재귀함수로 GCD 구하기
  def gcd(n, m):
      if n % m == 0:
          return m
      else:
          return gcd(m, n%m)
  
  def gcdlcm2(n, m):
      g = gcd(n, m)
      l = n*m // g
      return g, l
  
  print(gcdlcm2(3, 12))
  print(gcdlcm2(1071, 1029))
  
  """result
  [3, 12]
  [21, 52479]
  (3, 12)
  (21, 52479)"""
  ```

- 과일 개수 골라내기

  ```python
  basket_items = {'apples': 4, 'oranges': 19, 'kites': 3, 'sandwiches': 8}
  fruits = ['apples', 'oranges', 'pears', 'peaches', 'grapes', 'bananas']
  
  def fruits_checker(d):
      cnt = 0
      non_cnt = 0
      for k, v in d.items():
          if k in fruits:
              cnt += v
          else:
              non_cnt += v
      return cnt, non_cnt
  fruits_checker(basket_items)
  ```

- 절대값 함수 만들기

  절대값은 숫자(int, float)가 들어오면 절대값을 반환하고, 복소수(complex)가 들어오면 그 크기를 반환

  - 복소수 크기

    : 복소 평면이라는 x축이 실수부, y축이 허수부인 그래프에서 3+4j라면 (3,4)인 지점의 벡터거리(원점으로부터 직선 거리)가 복소수의 크기

  ```python
  # 방대승님
  def my_abs(x):
      return (x.real**2+x.imag**2)**0.5
  
  # 오재석님
  def my_abs(x):
      # conjugate는 켤레복소수를 제곱 제곱 -1 하여 반환
      return (x * x.conjugate()).real**0.5
  
  갓동주님 명세서 따르기 코드
  def my_abs(x):
      if type(x) == type(1j):
          return (x.real**2+x.imag**2)**0.5
      else:
          if x == 0:
              return x ** 2
          elif x < 0:
              return x * -1
          else:
              return x
  
  print(abs(3+4j), abs(-0.0), abs(-5))
  """result
  5.0 0.0 5"""
  ```


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
  
  # lambda 사용
  pattern = lambda n : "\n".join([str(i)*i for i in range(n+1) if i %2])
  print(pattern(9))
  print(pattern(6))
  ```

  