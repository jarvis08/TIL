# Recursive Function

- 재귀 함수 = 수학에서의 점화식

  자기 자신을 호출하여 순환 수행되는 것

  재귀가 아닌 함수 보다 프로그램의  크기를 줄이고 간단하게 작성 가능

  e.g., factorial(선형 재귀), 피보나치

- factorial(5)

  `5 -> 4 -> 3 -> 2 -> 1`

  `5 <- 4 <- 3 <- 2 <- 1`

- 피보나치

  `i >= 2` 일 때

  ```python
  def fibo(n):
      if n < 2:
          return n
      else:
          return fibo(n-1) + fibo(n-2)
  ```

- 결정론적 알고리즘

  Input과 로직이 같다면 항상 같은 Output

### Back Tracking

- Back Tracking이라는 것은 재귀를 통해 구현하는 것이며,

  재귀를 사용하지 않을 경우 사용자 정의 Stack을 구현하여 함수 호출을 직접 쌓아줘야 한다.

  - 하지만 그런 방식으로 굳이 Back Tracking을 구현하지 않는다.

  - 즉, `재귀 = BackTracking`이라 할 수 있으면서도,

    엄밀히 따지자면 완전히 동일한 것은 아니다.

---

## Memoization

- 피보나치 수열을 위처럼 수행한다면, 시간 복잡도 = `O(n**2)`

  피보나치를 DFS를 이용하여 구현하면, **Overlapping Subproblem**이 다수 발생

  - Overlapping Subproblem : 같은 작업을 여러번 수행

    e.g., fibo(4)의 경우 fibo(2), fibo(3)이 여러번 수행

- Memoization

  - '메모리에 넣기'를 의미하며, '기억해야 할 것' 이라는 memorandum에서 파생

  - 같은 작업이 반복됐을 때 이를 다시 수행하지 않으며,

    저장해뒀던 값을 불러오므로서 효율 증가

- Memoization을 사용하여 피보나치 구현하기

  ```python
  def fibo(n):
      global memo
      if n >= 2 and len(memo) <= n:
          memo.append(fibo(n-1) + fibo(n-2))
      return memo[n]
  # memo[0]과 memo[1]을 0과 1로 초기화
  memo = [0, 1]
  print(fibo(4))
  print(memo)
  """result
  3
  0, 1, 1, 2, 3"""
  ```