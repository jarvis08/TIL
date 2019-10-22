# Lambda

 **참고자료** : <https://offbyone.tistory.com/73>

- `lmbda (변수) : (계산식)`

  ```python
  func = lambda x : x ** 2
  print(func(2))
  ```

  ```python
  a = [1, 2]
  b = [3, 4]
  print(list(map(lambda x, y : x + y, a, b)))
  """result
  [4, 6]"""
  ```

  ```python
  foo = [2, 9, 27, 3, 4, 5]
  print(list(filter(lambda x: x % 3 == 0, foo)))
  """result
  [9, 27, 3]"""
  ```

  ```python
  from functools import reduce
  reduce(lambda x, y : x + y, [1, 2, 3, 4, 5])
  """result
  15"""
  ```