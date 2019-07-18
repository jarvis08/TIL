# SSAFY_Week3_Day3

 **참고자료** : ./50_SSAFY/8ython/notes/03.function.jpynb

---

- URL 편하게 만들기

  url 패턴을 만들어 문자열을 반환하는  `my_url` 함수를 만들어봅시다.

  영진위에서 제공하는 일별 박스오피스 API 서비스는 다음과 같은 방식으로 요청을 받습니다.

  ```
  기본 요청 URL : http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json?
  ```
  - key : 발급받은 키값(abc)
  - targetDt : yyyymmdd
  - itemPerPage : 1 ~ 10 **기본 10**

- URL 검증하기

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

---

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

---

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

  