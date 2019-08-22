# Graph 순회, DFS와 BFS)

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

- DFS 탐색 방법

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