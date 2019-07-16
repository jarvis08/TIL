# SSAFY_Week3_Day2

 **참고자료** : ./50_SSAFY/8ython/notes/02.control_of_flow.jpynb

---

## Scope

- global(바깥)에서 제어문의 내부(local) 변수 참조 불가
- 제어문 안에서는 바깥 변수를 참조 가능

---

## 제어문, Control of Flow

: 크게 반복문과 조건문으로 나뉘며, 순서도(Flow chart)로 표현이 가능

- **조건문**

  1. `if` 문은 반드시 일정한 참/거짓을 판단할 수 있는 `조건식`과 함께 사용, `if <조건식>:`

  2-1. `<조건식>`이 참인 경우 `:` 이후의 문장을 수행

  2-2. `<조건식>`이 거짓인 경우 `else:` 이후의 문장을 수행

  - 이때 반드시 **들여쓰기를** 유의!

    파이썬에서는 코드 블록을 자바나 C언어의 `{}`와 달리 `들여쓰기`로 판단

  - 앞으로 우리는 `PEP-8`에서 권장하는 `4spaces`를 사용

- **복수 조건문**

  2개 이상의 조건문을 활용할 경우 `elif <조건식>:`을 활용

- **조건 표현식, Conditional Expression**

  `true_value if <조건식> else false_value`

  : 한 줄 표현, 보통 다른 언어에서 활용되는 삼항연산자와 동일

  ```python
  # example_1
  num = int(input("숫자를 입력하세요 : "))
  value = num if num >= 0 else 0
  print(value)
  
  # example_2
  num = 2
  result = '홀수입니다.' if num % 2 == 1 else '짝수입니다.'
  print(result)
  ```

  ---

- **반복문**

  - **`while`** 문

    `while`문은 조건식이 참(True)인 경우 반복적으로 코드를 실행

    `while`문 역시 `<조건식>`이후에 `:`이 반드시 필요하며,

    이후 오는 코드 블록은 `4spaces`로 들여쓰기

  - **`for`** 문

    `for`문은 정해진 범위 내(시퀀스)에서 순차적으로 코드를 실행

    ```python
    for variable in sequence:
        line1
        line2
    ```

    `for`문은 `sequence`를 순차적으로 `variable`에 값을 바인딩하며, 코드 블록을 실행

    ```python
    # python에서의 if문과 for문은 block base scope가 아니다!!
    # 따라서 타 언어에서와 같이 i를 variable로 설정 시, 바깥의 i와 꼬일 수 있다.
    # 함수는 block base scope 처리
    ```

    ```python
    # string 중 모음 제거
    my_str = "Life is too short, you need python"
    li=[i for i in my_str if i not in 'aeiou']
    for i in li:
       print(i, end='')
    
    # 1~30 중 홀수만 출력
    odds = [num for num in range(1,31) if num % 2]
    print(odds)
    ```

    - **index**와 함께 `for`문 활용

      **`enumerate(iterable, start=0)`**를 활용한 추가적인 변수 활용

      : 열거 객체를 돌려주며, 반환된 이터레이터의 `__next__()` 메서드는 카운트와 iterable을 이터레이션 해서 얻어지는 값을 포함하는 튜플을 반환

      ```python
      # example_1
      lunch = ['짜장면', '초밥']
      for idx, menu in enumerate(lunch):
          print(menu)
          print(idx)
      """result
      짜장면
      0
      초밥
      1"""
      
      # example_2
      classroom = ['정의진', '김민지', '김건호', '김명훈']
      print(list(enumerate(classroom)))
      """result
      [(0, '정의진'), (1, '김민지'), (2, '김건호'), (3, '김명훈')]"""
      ```

    - **`dictionary`** 반복문 활용

      ```python
      friend = {'이름':'김병철', '여성':True, '주소':'수원', '전공':'System Management Engineering'}
      for key in friend:
          print(key, end=' : ')
          print(friend[key])
      
      for item in friend.items():
          print(item)
      
      for k, v in friend.items():
          print(k, v)
      
      for k in friend.keys():
          print(k)
      
      for v in friend.values():
          print(v)
      """result
      이름 : 김병철
      여성 : True
      주소 : 수원
      전공 : System Management Engineering
      ##############################
      ('이름', '김병철')
      ('여성', True)
      ('주소', '수원')
      ('전공', 'System Management Engineering')
      ##############################
      이름 김병철
      여성 True
      주소 수원
      전공 System Management Engineering
      ##############################
      이름
      여성
      주소
      전공
      ##############################
      김병철
      True
      수원
      System Management Engineering"""
      
      # 함수 사용
      blood_type = {"A": 4, "B": 2, "AB": 3, "O":1}
      print(f'총인원은 {sum(blood_type.values())}')
      ```

    - **`break`**, **`continue`**, **`else`**

      - `beak`

      - `continue`

        : `continue` 이후의 코드를 수행하지 않고 다음 요소를 선택해 반복을 계속 수행

        `continue`는 코드를 복잡하게 만드는 경향이 있어 꼭 필요한 경우가 아니고서는 잘 사용하지 않음

      - **`else`**

        : `else`문은 끝까지 반복문을 시행한 이후에 실행

        **`break`를 통해 중간에 종료되지 않은 경우**에만 실행

        ```python
        # else는 if문 뿐만 아니라 for문과도 사용 가능
        for i in range(3):
            print(i)
            if i == 100:
                print(f'{i}에서 break 걸림')
                break
        else:
            print("break가 안걸림")
        ```

        ```python
        # example
        numbers = [1, 5, 10]
        for num in numbers:
            if num == 3:
                print("True!")
                break
        else:
            print('False')
        """result
        False"""
        ```

---

## 함수, Function

- 복잡한 세상을 컴퓨터에게 단순하게 이해시키도록 하는 것이 사람이 할 일

  세상 문제들의 **`Complexity`**를 낮추기 위해 **`Abstraction`**(추상화 X, **요약** O)을 이용하는 것이 핵심

-  **`함수, Function`**은 **`Abstraction`**의 수단 중 하나

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

    (`return` 값이 없으면 None을 반환, i.g., `print()`)

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

- 함수의 **인수**

  : 함수는 `인자, parameter` 전달이 가능

  - 위치 인수

    : 함수는 기본적으로 인수를 위치로 판단

    ```python
    # 순서를 바꾸면 다른 값 return
    def cylinder(r, h):
        return r**2 * h * 3.14
    ```

- **기본 값, Default Argument Values**

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

- **키워드 인자, Keyword Arguments**

  : 키워드 인자는 직접적으로 **변수의 이름**으로 **특정 인자**를 전달

  ```python
  def greeting(age, name='john'):
      print(f'{name}은 {age}살입니다.')
      pass
  greeting(24, name='철수')
  ```

  ```python
  # print(*object, sep='', end='\n', file=sys.stdout, flush=False)
  # * : asterisk
  print('첫번째 문장')
  print('두번째 문장', end='_')
  print('세번째 문장', '마지막 문장', sep="/", end="끝!")
  """result
  첫번째 문장
  두번째 문장_세번째 문장/마지막 문장끝!"""
  ```

- **가변 인자 리스트**

  `print()`처럼 **정해지지 않은 임의의 개수의 인자**를 받기 위해서는 가변인자를 활용

  가변인자는 **`tuple` 형태**로 처리가 되며, `*`(asterisk)로 표현

  `def func(*args):`

  ```python
  def my_max(*nums):
      return max(*nums)
  ```

- **정의되지 않은 인자**들 처리하기

  정의되지 않은 인자들은 **`dict` 형태**로 처리되며, __`**`로 표현__

  **주로 `kwagrs`라는 이름**을 사용하며, `**kwargs`를 통해 인자를 받아 처리

  `def func(**kwargs):`

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

  

- **dictionary를 인자로 넘기기, unpacking arguments list**

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

  

  

  