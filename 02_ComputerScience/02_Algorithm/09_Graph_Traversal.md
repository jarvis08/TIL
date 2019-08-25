# Graph Traversal

## Graph

- 비선형구조 Graph

  그래프로 표현된 모든 자료를 빠짐 없이 검색하는 것이 중요

  - e.g., 미로

    벽을 만나면 갈 수 있는 방향이 제한됨

### Graph 표현 방법

- 인접 행렬, Adjacency Matrix

  - 가시적인 표현
- 필요한 메모리 resource가 크며, data 수가 많아질 경우 속도 저하를 야기
  
```python
  # 1-3, 1-4, 2-4, 3-6 형태의 연결
_input = [1, 3, 1, 4, 2, 4, 3, 6]
  adj = [[0]*(n_node+1) for _ in range(n_node+1)]
for i in range(int(len_input//2)):
      adj[_input[i*2]][_input[i*2+1]] = 1
    # 양방향 연결일 경우
      adj[_input[i*2+1]][_input[i*2]] = 1
```
  
- 인접 리스트, Adjacency List

  - 연결 개수에 비해 vertex의 개수가 적은 그래프에 용이
- 인접 리스트는 vertex 개수에 비례하는 메모리만을 차지하므로 인접 행렬 보다 효율적
  
```python
  graph = {
    'A': ['B'],
      'B': ['A', 'C', 'H'],
      'C': ['B', 'D', 'G'],
      'D': ['C', 'E'],
      'E': ['D', 'F']
  }
  ```

### Graph 순회

- 순회

  비선형 자료구조를 빠짐 없이 조사하는 완전 탐색 방법

- 순회의 종류

  - **깊이 우선 탐색(DFS, Depth First Search)**

    - Tree 순회에 사용
    - Stack 구조와 Visited 처리를 이용하여 탐색
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

## DFS의  응용

### AOV, Activity On Vertex

- 구성 요소

  - 정점 : Activity(작업)

  - 간선 : 작업간의 우선순위 관계를 나타내는 방향 그래프

    e.g, (1, 2), (1, 4), (2, 3) 의 연결이 있고, 좌측 값이 우측값의 선행 요소

  - 차수 : node에서의 경로 개수

    - 진입 차수, predecessor

      노드로 진입하는 경로 개수

    - 진출 차수, successor

      노드에서 진출하는 경로 개수

- 구현 방법

  1. 방법 1

     진입 차수가 0인 노드 부터 제거하며 기록

  2. 방법 2, **위상 정렬(Topolocal Sort) 및 DFS**를 이용

     - 진입 차수와 진출 차수의 방향을 역으로(진입이 진출로, 진출이 진입으로) 설정 후 DFS를 이용하여 탐색

     - 진출을 계속 하다가 더 이상 진출 할 정점이 없을 때,
     
       즉, Back 해야 할 때에 Visited를 기록
     
     - 방향성이 존재하여 진출의 전제조건(모든 진입이 완료)이 있기 때문에,
     
       **진출 방향을 역으로 설정하지 않으면 DFS**가 불가능

### AOE, Activity On Edge

> 참고 자료
>
> <https://bombofmetal.tistory.com/449>

- AOE의 목적

  Graph의 임계 경로(Critical Path)를 구하는 것

- 구성 요소

  - 정점 : 작업의 완료를 알리는 사건(event)
  - 간선 : 작업들의 선후관계와 작업에 필요한 시간을 표시

- 임계경로, Critical Path

  프로젝트 완료의 최소시간이며, 시작 정점에서 최종 정점까지의 **가장 긴 경로**의 길이

  - 자료구조에서의 수행시간은 최적의 경우가 아니라 **최악의 경우에 얼마만큼의 퍼포먼스를 보여주냐**에 있다.

    그러므로 한 프로젝트를 완료하기 위한 최소시간은 여러 시나리오 중 최악의 경우이다.

---

## BFS, Breadth First Search

### BFS 특징

- node에 소요 시간과 같은 가중치가 없을 때에는 미로 탐색을 수행 가능

  하지만 그렇지 않은 조건에서는 **순열**을 사용해야만 가능

- 상황에 따라 우선순위 대기열(Priority Queue) 사용

  Queue에 쌓을 때 원하는 순서대로 Sorting 후 Enqueue(PUSH)하여 검색 성능 개선

  Priority Queue는 힙(heap)을 사용하여 효과적으로 구현 가능

- Target Node를 발견할 때까지 모든 노드를 적어도 한 번은 방문하고 싶을 때 사용

- 최단 경로 구할 때 반드시 BFS를 사용해야 하는 경우 존재

  찾아진 경로들 중 가장 먼저 찾은 경로가 최단 경로

### 구현

1. 다음 깊이의 노드를 방문
2. 방문한 노드를 Dequeue
3. 방문한 노드의 다음 (깊이의) 노드들을 Enqueue, 방문 처리
   - 방문 기록 방법
     - Dequeue 할 때
     - Enqueue 할 때
4. 모든 노드들을 방문할 때 까지 1~3 과정을 반복

- 2-Dimention List에 노드 별 인접 행렬을 작성하여

  0과 1로 인접 여부를 판단하는 방법

  ```python
  # Graph를 2-D List로 표현했다면,
  # graph : 순회 할 graph 전체
  # start : Root 노드, 순회의 시작 노드
  def bfs(graph, start):
      visited = []
      queue = [start]
      while queue:
          node = queue.pop(0)
          for child in node_childs:
              visited.append(child)
              queue.append(child)
      return visited
  ```

- Dictionary를 이용하여 인접 노드를 표시

  ```python
  # 출처 : https://itholic.github.io/python-bfs-dfs/
  graph = {
      'A': ['B'],
      'B': ['A', 'C', 'H'],
      'C': ['B', 'D', 'G'],
      'D': ['C', 'E'],
      'E': ['D', 'F'],
      'F': ['E'],
      'G': ['C'],
      'H': ['B', 'I', 'J', 'M'],
      'I': ['H'],
      'J': ['H', 'K'],
      'K': ['J', 'L'],
      'L': ['K'],
      'M': ['H']
  }
  ```

  ```python
  def bfs(graph, start):
      visited = []
      queue = [start]
  
      while queue:
          node = queue.pop(0)
          if node not in visited:
              visited.append(node)
              # queue.extend(graph[node])
              queue += graph[node] - set(visited)
      return visited
  ```

- 가능한 모든 경로 구하기

  ```python
  def bfs_paths(graph, start, goal):
      queue = [(start, [start])]
      result = []
  
      while queue:
          n, path = queue.pop(0)
          if n == goal:
              result.append(path)
          else:
              for m in graph[n] - set(path):
                  queue.append((m, path + [m]))
      return result
  ```

  

---

## DFS와 BFS의 차이

### 메모리 사용량

- DFS는 재귀식 사용 시 Stack(컴퓨터 메모리) Overflow 발생 가능성 농후

  - C, C++ 등은 함수를 얉게(가볍게) 선언하면 재귀를 깊은 단계까지 사용이 가능

  - Python의 경우 system에서 메모리량이 아닌 함수 호출 개수로 설정

    `sys` library를 이용하여 설정을 변경하여 제한 범위를 작게 설정 가능

- BFS의 경우 메모리 사용량이 훨씬 작다

### 방문 처리 방식

- DFS는 이웃을 모두 처리 한 다음에 자기 자신을 방문 처리

  Stack에 자기 자신을 계속해서 쌓으며, 끝에 도달하면 역방향으로 처리하기 시작

- BFS의 경우 자기 자신을 방문처리 한 후 이웃을 탐색하며 Queue에 추가

### 처리 방향

- DFS의 경우 몇 단계 후에 타겟 노드를 탐색 하는지 알 수 없다

  특정 노드의 탐색 순서는 탐색 방향에 따라 크게 달라짐

- BFS는 가까운 노드들부터 차례대로 수행하므로 항상 일정한 순서 및 방향을 유지