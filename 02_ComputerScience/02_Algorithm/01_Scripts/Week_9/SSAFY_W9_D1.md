# SSAFY Week9 Day1

- deque, double-ended queue

  `appendleft()`

  `extendleft()`

  `popleft()`

  `rotate(int_n)` : 정수(음수 가능) 값 만큼 요소를 회전(순서 밀어내기)

  ```python
  import collections
  
  deq = collections.deque(['a', 'b', 'c'])
  deq.appendleft('d')
  print(deq)
  '''result
  deque(['d', 'a', 'b', 'c'])
  '''
  
  deq1 = collections.deque(['a', 'b', 'c', 'd', 'e'])
  deq1.rotate(-2)
  print('deq1 >>', ' '.join(deq1))
  '''result
  deq1 >> c d e a b
  '''
  ```

---

- In Python, `pydoc` as well as unit tests require modules to be importable. Your code should always check `if __name__ == '__main__'`before executing your main program so that **the main program is not executed when the module is imported**.

  ```python
  def main():
      ...
  
  if __name__ == '__main__':
      main()
  ```

---

## Stack2

### 계산기

실제 System Stack에서는 계산식이 들어오면 **후위 표기식으로 변환하여 계산**

- `4 + 3 * 5`

  - 토큰 5개

    - Token Analyzing

      프로그래밍의 Token :: 의미가 있는 가장 작은 단위

  - 피연산자 3개

  - 연산자 2개

    - 단항 연산자 0개
    - 이항 연산자 2개
    - 중위 표기법

- 식의 표기법

  - 전위 표기법

  - 중위 표기법, Infix Notation

    연산자를 피연산자의 가운데 표기하는 방법

  - 후위 표기법, Postfix Notation

    연산자를 피연산자 뒤에 표기하는 방법

- 중위 표기식을 후위 표기식으로 변환하기

  1. 수식의 각 연산자에 대해 우선순위에 따라 괄호를 사용하여 다시 표현
  2. 각 연산자를 그에 대응하는 오른쪽괄호의 뒤로 이동
  3. 괄호 제거

  ```
  e.g., A*B-C/D
  1. ((A*B)-(C/D))
  2. ((A B)* (C D)/)-
  3. AB*CD/-
  ```

- Stack 사용하여 **중위 표기식을 후위 표기식으로** 변환하기

  - Stack에 계속해서 **연산자와 괄호**를 `push()`, 피연산자는 바로 출력
  - 괄호가 끝나는 시점마다 괄호 사이의 연산자를 `Stack.top`부터 출력
  - 현재 순서의 연산자가 `Stack.top` 보다 우선순위가 높을 시 `Stack.top`을 출력 후 `Stack.push()`
  - 현재 순서의 연산자가 `Stack.top` 보다 우선순위가 같거나 낮으면 바로 출력

  ```
  (6 + 5 * (2 - 8) / 2)
  >> 6528-*2/+
  ```

- Stack을 사용하여 **후위 표기식 계산**하기

  - 후위 표기식으로의 변환 알고리즘과 반대로 Stack에 **피연산자**를 `push()`
  - 현재 순서가 연산자일 경우
    1. `Stack.top`을 `pop()`
    2. `Stack.top (연산자) St ack.top-1`의 **계산 결과**를 `push()`

### 백트랙킹, Back-Tracking

해를 찾는 도중 **막히면**, 즉 해가 아니면 되돌아가서 다시 해를 찾는 기법이다.

- 재귀 함수를 사용하는 DFS와 동일한 동작 방식

- 최적화(Optimization) 문제와 결정(Decision) 문제를 해결 가능

- 결정(Decision) 문제

  문제의 조건을 만족하는 해가 존재하는가의 여부를 O/X로 답하는 문제

  - 미로에서의 길 찾기

    Stack을 `push()`하며 경로를 기록해 가지만,

    더이상 진행할 수 없을 시 `pop()`을 하며 되돌아감

  - n-Queen 문제

    체스판의 여러 Queen을 배정하는 방법 탐색

  - Map Coloring

  - 부분 집합의 합(Subset Sum) 문제

- DFS와의 차이점

  - Back-Tracking의 경우 **Prunning**(가지치기)이 목적

    어떤 노드에서 출발하는 경로가 해결책으로 이어질 것 같지 않으면 더 이상 그 경로를 따라가지 않음으로써 시도의 횟수를 줄임

  - DFS는 모든 경로를 추적(순회)하는 것이 목적

  - DFS는 N!의 시간 복잡도를 가지며,

    Back-Tracking의 경우에도 최악의 경우 Exponential(지수함수의) Time

- Prunning

  어떤 노드의 유망성을 점검한 후 유망(Promising)하지 않다고 결정되면 노드의 부모 노드로 되돌아가서 다음 자식 노드를 방문

  1. 상태 공간 트리의 DFS를 실시
  2. 노드의 유망성을 점검
  3. 유망하지 않다면 부모 노드로 돌아가서 탐색 재개

- 일반적인 Back-Tracking

  ```python
  def checknode(vertex):
      """promising
      # n-Queen의 경우, 직선/대각선에 다른 Queen이 없으면 promising
      DFS의 경우 안되는 모든 경우의 수를 계산하지만,
      Back-Tracking의 경우 안되는 경우 하나라도 있으면,
      그로부터 파생되는 경우들을 prunning한다.
      """
      if promising(vertex):
          # solution : 과정이 끝났는가를 검사
          if solution in vertex:
              return solution
          # solution이 완성되지 않았다면, 자식 노드를 탐색
          else:
              for child in vertex:
                  checknode(child)
  ```

- 4-Queen 문제(4x4 체스판, 4개 Queen)를 풀 때,

  DFS와 Back-Tracking 탐색 경우의 수 비교

  - Default DFS :: 155 nodes
  - Back-Tracking :: 27 nods

- 멱집합, Powerset

  `powerset.py` 참고

  ```python
  # a : 생성 과정의 부분 집합
  # k : initial depth
  # target : 목표 부분집합의 개수
  def backtrack(a, k, target):
      c = [0] * MAX_CANDIDATES
      
      if k == target:
          process_solution(a, k)
      else:
          k += 1
          n_cands = make_candidates(a, k, target, c)
          for i in range(n_cands):
              a[k] = c[i]
              backtrack(a, k, target)
          
  def make_candidates(a, k, target, c):
      c[0] = True
      c[1] = False
      return = 2
  
  def process_solution(sub, depth):
      for i in range(1, k):
          if a[i] == True:
              print(i)
              
  MAXCANDIDATES = 100
  NMAX = 100
  a = [0] * NMAX
  # 3개의 원소를 가지는 powerset
  backtrack(a, 0, 3)
  ```

- Back-Tracking 이용하여 Permutation 구현

  구성 요소만 다를 뿐, 동작 방식은 멱집합 구하기와 동일

  ```python
  def backtrack(a, k, target):
      global MAXCANDIDATES
      c = [0] * MAXCANDIDATES
      
      if k == target:
          for i in range(1, k+1):
              print(a[i], end=" ")
          print()
      else:
          k += 1
          n_cands = construct_candidates(a, k, target, c)
          for i in range(n_cands):
              a[k] = c[i]
              backtrack(a, k , target)
              
  def construct_candidates(a, k, target, c):
      in_permu = [False] * NMAX
      
      for i in range(1, k):
          in_permu[a[i]] = True
          
      n_cands = 0
      for i in range(1, target+1):
          if in_perm[i] == False:
              c[n_cands] = i
              n_cands += 1
      return n_cands
  ```

### 분할정복

문제를 나누어서 부분적으로 해결

- 설계 전략

  - 분할, Divide

    해결 할 문제를 여러 개의 작은 부분으로 분할

  - 정복, Conquer

    나누어 놓은 작은 문제를 각각 해결

  - 통합, Compoine

    필요 시, 해결된 답을 통합

- `n^k `을 O(log2n)의 시간 복잡도로 구현하기

  ```python
  def power(Base, Exponent):
      if Exponent==0 or not Base==0:
          return 1
      if Exponent % 2 == 0:
          NewBase = Power(Base, Exponent/2)
          return NewBase * NewBase
      else:
          NewBase = Power(Base, (Exponent-1)/2)
          return (NewBase * NewBase) * Base
  ```

- **퀵 정렬**, Quick Sort

  주어진 배열을 두 개로 분할하고, 각각을 정렬

  - 최악의 경우 O(n^2)의 시간복잡도를 가지지만,

    평균적으로는 O(nlogn)이기 때문에 가장 많이 사용됨

  - 합병 정렬과의 차이

    1. 합병 정렬은 그냥 두 부분으로 나누는 반면,

       퀵 정렬은 분할 할 때 기준 아이템(pivot item)을 중심으로 pivot 보다 작은 것은 왼편, 큰 것은 오른편에 위치시키는 규칙이 존재

    2. 각 부분의 정렬이 끝난 후 합병 정렬은 '합병'이라는 후처리 작업이 필요

       퀵정렬은 아무런 작업이 필요 없다.

  - 코드 설계

    1. pivot 값을 기준으로,

       - 좌측에서는 pivot 보다 큰 값을 탐색
       - 우측에서는 pivot 보다 작은 값을 탐색

    2. 구해 진 큰 값과 작은 값의 위치를 교환

    3. 만약 L<R이 성립되는 조건아래 교환할 값들을 찾지 못한다면,

       L/R이 위치한 값과 Pivot의 위치를 바꿈

  ```python
  def quickSort(a, begin, end):
      # 시작 위치가 끝 위치보다 작은 경우만 시행, 아니라면 작업의 필요가 없다.
      if begin < end:
          # pivot을 기준으로 좌/우로 분할하여 따로 계산
          p = partition(a, begin, end)
          quickSort(a, begin, p-1)
          quickSort(a, p+1, end)
          
  
  # pivot 값을 기준으로,
  # 좌측에서는 pivot 보다 큰 값을 탐색
  # 우측에서는 pivot 보다 작은 값을 탐색
  # 두 값의 위치를 교환
  def partition(a, begin, end):
      pivot = (begin + end) // 2
      L = begin
      R = end
      while L < R:
          while(a[L] < a[pivot] and L < R):
              L += 1
          while(a[R] >= a[pivot] and L < R):
              R -= 1
          # 만약 L<R이 성립되는 조건아래 교환할 값들을 찾지 못한다면,
          # L/R이 위치한 값과 Pivot의 위치를 바꿈
          if L < R:
              if L == pivot:
                  pivot = R
              a[L], a[R] = a[R], a[L]
      a[pivot], a[R] = a[R], a[pivot]
      return R
              
  ```

  

