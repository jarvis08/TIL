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

## Exceptions

