# Graph 순회, DFS와 BFS

- 비선형구조 Graph

  그래프로 표현된 모든 자료를 빠짐 없이 검색하는 것이 중요

  - e.g., 미로

    벽을 만나면 갈 수 있는 방향이 제한됨

- 무방향 그래프 탐색 방법

  - 인접 행렬 그리기

    - 장점

      가시적

    - 단점

      표현하는데 resource가 x2

      data 수가 많아질 경우 속도 저하를 야기

  - 인접 리스트

- 비선형 자료구조를 조사하는 방법 = **순회**

  - **깊이 우선 탐색(DFS, Depth First Search)**

    Stack 구조와 Visited 처리를 이용하여 탐색

    - 탐색하며 나아가던 중 돌아올 수 있어야 함
    - Visitied를 참고하여 탐색한 곳을 다시 탐색하는 것을 방지

  - **너비 우선 탐색(BFS, Breadth First Search)**

    Queue 구조와 Visited 처리를 이용하여 탐색

---

## DFS, Depth First Search

### DFS 특징

- 때에 따라서는 A,B,C,D 노드처럼 간선을 줄 수도, 미로와 같은 형태로 배열을 줄 수도 있다.
- 재귀를 사용하게 되면 사용자 정의 Stack이 아닌 System Stack을 이용하는 것이며,
- 재귀를 사용하므로 Back Tracking 방식의 연산을 하게 된다.
- 사용 구조
  - 스택
  - 반복
  - 재귀

### DFS 탐색 방법

1. 이동하면서 Stack에 이동할 수 있는 모든 node를 기록
2. 방문한 곳은 다시 Stack에서 pop(방문하지 않은 곳은 남겨둠)
3. Visited List에 방문한 곳을 기록
4. 끝에 닿았을 경우 다시 돌아오는 길에 1,2,3번 과정을 똑같이 반복하며 첫 위치까지 복귀
5. Stack에 남아있는 요소를 검사하고, 남아있을 경우 Visited List와 비교하여 확인

```python
stack = []
visited = []
def DFS(v):
	push(s, v)
    # stack에 data가 있는한 계속 시행
    while not isEmpty(s):
        v = pop(stack)
        if not visited[v]:
            # 방문했음을 표시
            visit(v)
            # adjacency : 인접
            for w in adjacency(v):
                if not visited[w]:
                    push(s, w)
```

```python
# 재귀 사용
def dfs_recursive(G, v):
    # 방문 했음을 표시
    visited[v] = True
    for w in adjacency(G, v):
        if not visited[w]:
            dfs_recursive(G, w)
```

### DFS의  응용, 위상 정렬

- 위상 정렬(AOV, Activity On Vertex)

  정점이 Activity, 작업을 나타내고, 간선이 작업간의 우선순위 관계를 나타내는 방향 그래프

  e.g, (1, 2), (1, 4), (2, 3) 의 연결이 있고, 좌측 값이 우측값의 선행 요소

- 가능한 경로 두 가지
  
  1423
  
  4123

- 차수 : node에서의 경로 개수

  - 진입 차수, predecessor

    노드로 진입하는 경로 개수

  - 진출 차수, successor

    노드에서 진출하는 경로 개수

- 풀이 방법

  1. 방법 1

     진입 차수가 0인 노드 부터 제거하며 기록

  2. 방법 2, DFS/Visited

     - 진입 차수와 진출 차수의 방향을 역으로(진입이 진출로, 진출이 진입으로) 설정 후 DFS를 이용하여 탐색

     - 진출을 계속 하다가 더 이상 진출 할 정점이 없을 때,
     
       즉, Back 해야 할 때에 Visited를 기록
     
     - 방향성이 존재하여 진출의 전제조건(모든 진입이 완료)이 있기 때문에,
     
       **진출 방향을 역으로 설정하지 않으면 DFS**가 불가능

### DFS의  응용, 임계 경로

- 임계 경로(AOE, Activity On Edge)

---

## BFS, Breadth First Search

### BFS 특징

- 무방향 그래프이며 가중치가 없을 때에는 미로 탐색을 수행 할 수 있다.

  하지만 그렇지 않은 조건에서는 순열을 사용해야만 가능

- 또한 Queue에 쌓을 때 원하는 순서대로 Sorting하여 PUSH 할 수 있으므로, 때에 따라서는 성능을 향상 시킬 수 있다.

- 최단 경로 구할 때 반드시 BFS를 사용해야 하는 경우 존재

  - Queue와 Iteration을 사용

### DFS와 BFS의 차이

- 메모리 사용량

  - DFS는 재귀식 사용 시 Stack(컴퓨터 메모리) Overflow 발생 가능성 농후

    - C, C++ 등은 함수를 얉게(가볍게) 선언하면 재귀를 깊은 단계까지 사용이 가능

    - Python의 경우 system에서 메모리량이 아닌 함수 호출 개수로 설정

      `sys` library를 이용하여 설정을 변경하여 제한 범위를 작게 설정 가능

  - BFS의 경우 메모리 사용량이 훨씬 작다

- 방문 처리 방식

  - DFS는 이웃을 모두 처리 한 다음에 자기 자신을 방문 처리

    Stack에 자기 자신을 계속해서 쌓으며, 끝에 도달하면 역방향으로 처리하기 시작

  - BFS의 경우 자기 자신을 방문처리 한 후 이웃을 탐색하며 Queue에 추가

- 처리 방향

  - DFS의 경우 몇 단계 후에 타겟 노드를 탐색 하는지 알 수 없다

    특정 노드의 탐색 순서는 탐색 방향에 따라 크게 달라짐

  - BFS는 가까운 노드들부터 차례대로 수행하므로 항상 일정한 순서 및 방향을 유지