# Errors & Exceptions

 **참고자료** : ./50_SSAFY/8ython/notes/06.erros.ipynb

---

## Errors

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

---

## 예외처리, Exceptions

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

  ------

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

  ------

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

  ------

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

  ------

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

  ------

- `raise`

  예외 발생시키기

  ```python
  # ValueError와 같은 에러 유형도 함수이다!
  # 따라서 ('메세지')를 부여하여 에러와 함께 특정 메세지 출력 가능
  num = input('숫자 :: ')
  if num.isalpha():
      raise ValueError('숫자를 입력하세요')
  ```

  ------

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