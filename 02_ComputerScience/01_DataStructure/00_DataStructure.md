# About Data Structure

---

- ADT, Abstract Data Type
  - 자료

  - 연산

    자료구조 별 논리적 구조를 유지한 채 데이터를 다룸

    - 삽입

    - 삭제

    - **순회**, Traversal

      주어진 자료 구조의 모든 element를 빠짐 없이 조회할 수 있는 방법

---

## 자료 구조

### 선형

> element와 element의 관계가 1:1
>
> 자기 자신에서 이어질 수 있는 경우의 수 = 1개
>
> 2차원일 경우, 같은 차원 당 경우의 수 = 1개
>
> 물리적 구조 == 논리적 구조

- 배열, Array
  - **일정한 자료형의 변수들**을 하나의 이름으로 열거하여 사용하는 자료구조

- 리스트, List

  탐색기와 같은 개념으로, 비어있는 공간이 존재 하지 않아 삭제 시 리스트 크기 축소

  삽입/삭제가 빈번하게 발생하는 자료에 적합

- 스택, Stack

- 큐, Queue

- 데크, Deck

### 비선형

> element와 element의 관계가 1:N

- **트리, Tree**

  - 트리의 표현
    - 1차 배열
    - 리스트
    - 스태틱 링크드 리스트
  - 트리의 순회
    - Pre-Order
    - In-Order
    - Post-Order
    - DFS, BFS를 사용할 수 있지만 닭 잡는데 소 잡는 칼을 쓰는격
  - 트리의 종류
    - 이진 트리
    - 이진 탐색 트리
    - AVL
    - B
    - 트라이
    - 허프만 트리
    - 아호코라식 트리
    - 세그먼트 트리

- **그래프, Graph**

  Tree 또한 방향이 없는, 무방향의 Graph

  cycle이 가능하며, 방향성이 존재할 수 있다.

  - 그래프의 표현

    - 인접 행렬

    - 인접 리스트

      메모리 소요가 크며, (스태틱) 링크드 리스트를 사용

  - 그래프의 순회

    - **DFS**

      깊이 우선 탐색

      막다른 길이면 되돌아옴

      Stack을 사용

      재귀형식의 경우 DFS 형식으로 호출

      Back Tracking이라고도 불림

    - **BFS**

      Queue를 사용하며, 가까운 것부터 탐색

  - 그래프의 종류

    - MST, Minimum Static Tree

    - 최단 경로

      경로 별 cost 계산

      - 다익스트라 알고리즘
      - TSP, Traveling Salesman Problem
      - 플로이드

    - AoV

    - AoE

    - 네트웍

    - 기하

