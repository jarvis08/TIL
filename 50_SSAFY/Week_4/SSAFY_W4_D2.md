# SSAFY_Week4_Day2

 **참고자료** : ./50_SSAFY/8ython/notes/06.errors.ipynb

---

## 예외처리

- `try`, `except`

  - `try`절을 실행
  - 예외가 발생되지 않으면, `except`없이 실행이 종료
  - 실행 중도에 지정한 예외 사항 발생시, 남은 부분을 수행하지 않고 `except` 실행

  ```python
  try:
      codeblock1
  # 예외 내용은 변수의 특정 값일 수 있으며,
  # Error 내용으로 지정할 수 있다
  # i.g., ValueError
  except 예외 내용:
      codeblock2
  ```

  ---

- 복수 예외 처리

  두 가지 예외를 모두 처리 가능

  ```python
  try:
      codeblock1
  except (예외1, 예외2):
      codeblock2
  ```

  ```python
  # 예시) 100을 사용자가 입력한 값으로 나눈 후 출력하는 코드
  try:
      var = input('숫자를 입력')
      print(100/int(var))
  except ValueError:
      print('바보야 숫자 입력하라니까')
  except ZeroDivisionError:
      print('0이 아닌 숫자를 입력하세요.')
  except:
      print('뭘 이상하게 했는지는 모르곘지만, 에러 났음')
  ```

  - **에러는 순차적으로 수행되므로, 가장 작은 범주부터 시작해야 한다!**

    ```python
    try:
        var = input('숫자를 입력 : ')
        print(100/int(var))
    except Exception:
        print('뭘 이상하게 했는지는 모르곘지만, 에러 났음')
    except ValueError:
        print('바보야 숫자 입력하라니까')
    except ZeroDivisionError:
        print('0이 아닌 숫자를 입력하세요.')
    
    """result
    숫자를 입력 : 0
    뭘 이상하게 했는지는 모르곘지만, 에러 났음"""
    # ZeroDivisionError이지만, 앞에서 걸러져버림
    ```

  ---

- 에러 문구 처리

  에러 문구를 print, 전달 할 수 있다.

  ```python
  try:
      num_list = [1, 2, 3]
      print(num_list[4,4])
  except Exception as err:
      print(err)
  """result
  list indices must be integers or slices, not tuple"""
  ```

  ---

- `else`

  에러가 발생하지 않는 경우 수행

  ```python
  try:
      codeblock1
  except 예외:
      codeblock2
  else:
      codeblock3
  ```

  ```python
  try:
      num_list = [1, 2, 3]
      num_list[2]
  except:
      print('에러 났어요')
  else:
      print(num_list[2])
  """result
  3"""
  ```

  ---

- `finally`

  예외 발생 여부에 상관 없이, 반드시 수행해야하는 문장

  ```python
  try:
      codeblock1
  except 예외:
      codeblock2
  finally:
      codeblock3
  ```

  ```python
  try:
      students = {'john':'cs', 'jaeseok':'math'}
      students['minji']
  except KeyError as em:
      print(f'{em} 는 딕셔너리에 없는 키입니다.')
  finally:
      print('곧 쉬는 시간. 조금만 힘내')
  
  """result
  'minji' 는 딕셔너리에 없는 키입니다.
  곧 쉬는 시간. 조금만 힘내"""
  ```

  ---

- `raise`

  예외 발생시키기

  ```python
  # ValueError와 같은 에러 유형도 함수이다!
  # 따라서 ('메세지')를 부여하여 에러와 함께 특정 메세지 출력 가능
  num = input('숫자 :: ')
  if num.isalpha():
      raise ValueError('숫자를 입력하세요')
  ```

  ---

- `assert`

  보통 상태를 검증하는데 사용되며, 무조건 `AssertionError` 발생

  ```python
  assert 'Boolean expression', 'error message'
  ```

  'Boolean expression'의 검증식이 거짓일 경우 `AssertionError` 발생

  `raise`는 **항상 예외**를 발생시키며, `assert`는 **지정 예외**만 발생한다는 점에서 차이

  ```python
  def my_div(num1, num2):
      assert type(num1) == int and type(num2) == int, '정수가 아닙니다.'
      try:
          result = num1 / num2
      except ZeroDivisionError as err:
          print(f'{err} 오류가 발생했습니다.')
      else:
          return result
  my_div(4,2)
  """result
  2.0"""
  ```

  _보통 예외 처리 용이 아닌, TDD 방법론과 같이 Test를 할 때 사용_

---

 **참고자료** : ./50_SSAFY/8ython/notes/07.OOP_basic.ipynb

## 객체 지향 프로그래밍, Object-Oriented Programming( OOP)

- 절차적(Procedual) 프로그래밍

  순서도(Flow Chart)를 그릴 수 있는 프로그래밍

  프로그램 규모가 커지면, 유지 보수가 어려움

- 객체 지향 프로그래밍

  - 이론으로만 존재했던 OOP를 smalltalk에서 처음으로 구현

  - (언어학+철학)에서 시작된 개념

    세상을 있는 그대로 표현할 수 있도록 사람처럼 묘사하는, 서술적인 사고 방식(주어 + 동사)을 구현하고자 시작

  - (주어 + 동사)는 (Object + Predicate)의 개념으로 발전

  - 따라서 Object를 '객체' 보다는 '사물'로 해석하는 것이 이해하기 쉬우며,

    _객체 지향 프로그래밍 = 사실적인 프로그래밍_ 이라고 해석할 수 있다.

  ```python
  # Object + Predicate
  # 'hello' + islower()
  # 'hello'를 islower() 해라
  'hello'.islower()
  ```

<wikipedia - 객체지향 프로그래밍>

> 객체 지향 프로그래밍(영어: Object-Oriented Programming, OOP)은 컴퓨터 프로그래밍의 패러다임의 하나이다. 객체 지향 프로그래밍은 컴퓨터 프로그램을 명령어의 목록으로 보는 시각에서 벗어나 여러 개의 독립된 단위, 즉 "객체"들의 모임으로 파악하고자 하는 것이다. 각각의 객체는 메시지를 주고받고, 데이터를 처리할 수 있다.
>
> 명령형 프로그래밍인 절차지향 프로그래밍에서 발전된 형태를 나타내며, 기본 구성요소는 다음과 같다.

- 클래스, Class

  - 같은 종류(또는 문제 해결을 위한)의 집단에 속하는 **속성(attribute)**과 **행위(behavior)**를 정의한 것으로 객체지향 프로그램의 기본적인 사용자 정의 데이터형(user define data type)이라고 할 수 있다

  - 클래스는 프로그래머가 아니지만 해결해야 할 문제가 속하는 영역에 종사하는 사람이라면 사용할 수 있고, 다른 클래스 또는 외부 요소와 독립적으로 디자인하여야 한다.

    ---

    우리는 처음 나무라는 식물을 배울 때 위키피디아로 배우지 않는다.
    이것도, 그것도, 저것도 나무라는 것을 학습해가며 나무들의 **공통점** 및 **체계**를 익힌다.
    OOP에서는 그 **분류 체계**를 **`class`**라고 칭하며, Object들을 **구조화** 한다.

  ---

- 인스턴스, Instance

  - 클래스의 인스턴스/객체(실제로 메모리상에 할당된 것)이다.

  - 객체는 자신 고유의 속성(attribute)을 가지며 클래스에서 정의한 행위(behavior)를 수행할 수 있다.

  - 객체의 행위는 클래스에 정의된 행위에 대한 정의(메서드)를 공유함으로써 메모리를 경제적으로 사용한다.

  - `instanciate` : instance를 만드는 작업

    ---

    나무 `class`에 속하는 여러가지 나무들을 `instance`라고 한다.

  ---

- 속성, Attribute

  - 클래스/인스턴스 가 가지고 있는 속성(값)

    ---

    `class` 나무

    `instance` 바오밥 나무

    `attribute` 나무 재질

  ---

- 메서드, Method
  - 클래스/인스턴스 가 할 수 있는 행위(함수)

| class / type | instance                 | attributes       | methods                                |
| ------------ | ------------------------ | ---------------- | -------------------------------------- |
| `str`        | `''`, `'hello'`, `'123'` | _                | `.capitalize()`, `.join()`, `.split()` |
| `list`       | `[]`, `['a', 'b']`       | _                | `.append()`, `reverse()`, `sort()`     |
| `dict`       | `{}`, `{'key': 'value'}` | _                | `.keys()`, `.values()`, `.items().`    |
| `int`        | `0`, `1`, `2`            | `.real`, `.imag` |                                        |