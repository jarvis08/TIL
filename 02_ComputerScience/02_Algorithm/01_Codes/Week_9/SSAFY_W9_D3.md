# SSAFY Week9 Day3

## Code Review

- 후위 표기법 계산

  ```python
  # string으로 되어 있는 숫자를 비교하면 ASCII Code로 전환하여 비교
  if '3' > '0':
      print(True)
  """rsult
  True"""
  ```

  

- n-Queen.py

  - 문제 초점

    - 순서 X : 조합
    - 순서 O : 순열

  - 순열로 풀어야 하며, 경우의 수는 총 6가지

  - 구현 방법

    - visited = [0 , 0, 0]

      value = [0 , 0, 0]

      row, col 별로 하나의  값만 갖도록, visited를 3개 열로 설정

    - sum값이 이전까지의 max보다 큰지 계속해서 비교

      비교가 끝난 이후에는 visited 값을 0으로 다시 변환

      첫번째 index는 유지하며, 2개를 모두 탐색한 이후에는 처음까지 0

    - Prunning

      최소값보다 큰 값이 더해진다면 언제든 나갈 수 있도록 `if/return` 작성

      따라서 재귀 함수의 parameter에 최소값이 존재

  - `visited`를 Binary를 사용하여 구현하는 방법도 존재 한다
  - 최소값을 빠르게 찾기 위해 DFS가 아닌 BFS를 통해 계속해서 최소값을 탐색하는 방법도 존재
    - 탐색 할 때 계속해서 Greedy를 기록

---

## Queue

### 기본적인 Queue

- FIFO, First in First out

  삽입한 순서대로 원소가 저장되며, 가장 먼저 삽입된 원소가 가장 먼저 삭제

  - 머리, Front

    가장 첫 번째 값의 앞 index를 가리켜야함

    - 따라서 Queue를 생성하고 Enqueue를 진행해도 Dequeue가 발생하기 전 까지 front는 `-1`을 유지

    - `front == rear`일 때 empty로 취급하며,

      `rear = front + 1 일 때 1개의 element가 있음

  -  꼬리, Rear

  - 삽입, Enqueue
  - 삭제, Dequeue
  - isEmpty()
  - isFull()

  - createQueue

    ```python
    front = rear = -1
    ```

### 연결 큐, Linked Queue

단순 연결 리스트(Singly Linked List)를 활용한 Queue

- 큐의 원소

  단순 연결 리스트의 노드

- 큐의 원소 순서

  노드의 연결 순서, 링크로 연결되어 있음

- front

  첫 번째 노드를 가리키는 링크

- rear

  마지막 노드를 가리키는 링크

- 상태 표현

  - 초기 상태, 공백 상태

    `front = rear = null`

- Linking 순서

  `[data-field, link-field]`

  - Enqueue

    1. 새로운 linked list 공간을 할당 받음
    2. 기존 link-field의 null값 대신 새로운 data-field 주소를 기입
    3. rear를 새로운 data-field 공간의 주소로 변경

  - Dequeue

    1. old가 삭제 할 노드를 가리키도록 저장

    2. front의 data-field 값을 저장

    3. front의 link-field를 front로 갱신

    4. 노드 삭제

       Garbage Collector가 알아서 삭제한 주소의 field를 정리

       (C/C++의 경우 직접 처리)

### Linked Queue 구현

### 우선순위 큐, Priority Queue

FIFO가 아닌, Dequeue의 순서를 부여

- 적용 분야(큐잉 시스템, Queueing System)
  - 시뮬레이션 시스템
  - 네트워크 트래픽 제어
  - OS의 Task Scheduling

### 버퍼, Buffer

Data를 처리 장치들이 있을 때, 장치들 간의 속도 차이를 극복하기 위해 사용

데이터를 한 곳에서 다른 한 곳으로 전송하는 동안 일시적으로 그 데이터를 보관하는 메모리 영역

- 버퍼링

  버퍼를 활용하는 방식이며, 버퍼를 채우는 동작을 의미

  e.g., 키보드 꾹 누를때 삐삐삐삐삐 비프음이 나는건 버퍼가 가득 찼음을 의미

- 버퍼의 자료 구조

  - 일반적으로 입출력 및 네트워크와 관련된 기능에 이용
  - 순서대로 입력/출력/전달되어야 하므로 FIFO 방식의 자료구조인 Queue를 활용

---

## BFS, Breadth First Search

- Vertex의 인접 vertex들을 모두 탐색한 이후, 같은 layer의 vertex에서 같은 작업을 수행

  탐색한 인접 vertex들 중 하나의 vertex에서 다시 이웃 vertex들을 탐색

  - DFS 처럼 layer들을 전진이 불가능 할 때 까지 탐색해 들어가는 것이 아니라,

    layer 별로 clear하며 탐색