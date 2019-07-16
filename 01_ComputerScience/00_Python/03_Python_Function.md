# Python_Function

 **참고자료** : ./50_SSAFY/8ython/notes/03.funtion.jpynb

---

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

  

  

  