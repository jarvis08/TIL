# Back-Tracking

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

### 4-Queen

- 4-Queen 문제(4x4 체스판, 4개 Queen)를 풀 때,

  DFS와 Back-Tracking 탐색 경우의 수 비교

  - Default DFS :: 155 nodes
  - Back-Tracking :: 27 nods

### 멱집합, Powerset

`Powerset.py` 참고

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

### Permutation 구현

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

### Target Subset Search

특정 개수의 부분집합 구성을 탐색하는 과정

```python
def construct_candidates(a, k, input, c):
    c[0] = True
    c[1] = False
    return 2

def process_solution(a, k):
    sum = 0
    for i in range(1, 11):
        if a[i] == True:
            sum += i
    if sum == 10:
        for i in range(1, 11):
            if a[i] == True:
                print(i, end = ' ')
        print()

def backtrack(a, k, input):
    c = [0] * MAXCANDIDATES

    if k == input:
        process_solution(a, k)  # 답이면 원하는 작업을 한다
    else:
        k += 1
        ncandidates = construct_candidates(a, k, input, c)
        for i in range(ncandidates):
            a[k] = c[i]
            backtrack(a, k, input)

MAXCANDIDATES = 100
NMAX = 100
a = [0] * NMAX
backtrack(a, 0, 10)


# def backtrack(a, k, sum):
#     global cnt
#     cnt += 1
#     if k == N:
#         if sum == 10:
#             for i in range(1, 11):
#                 if a[i] == True:
#                     print(i, end=' ')
#             print()
#     else:
#         k += 1
#         # if sum + k <= 10 :
#         #     a[k] = 1; backtrack(a, k, sum + k)
# 
#         a[k] = 1; backtrack(a, k, sum + k)
#         a[k] = 0; backtrack(a, k, sum)
# 
# N = 10
# a = [0] * (N + 1)
# 
# cnt = 0
# backtrack(a, 0, 0)
# print("cnt : ", cnt)
```

