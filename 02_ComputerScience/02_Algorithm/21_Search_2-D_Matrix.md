# 2차원 배열

- 1차원 list를 묶어놓은 list

- 2차원 이상의 다차원 list는 차원에 따라 index를 선언

- 3차원 list의 선언 : 세로 길이(행 개수), 가로 길이(열 개수)를 필요로 함

- Python에서는 데이터 초기화를 통해 변수선언과 초기화가 가능

- 배열 순회

  n x m 배열의 n*m 개의 모든 원소를 빠짐 없이 조사하는 방법

  - **행 우선 순회**

    ```python
    # l : list
    # i : 행
    # j : 열
    for i in range(len(l)):
        for j in range(len(l[i])):
            l[i][j]
    ```

  - **열 우선 순회**

    ```python
    # l : list
    # i : 행
    # j : 열
    for j in range(len(l[0])):
        for i in range(len(l)):
            l[i][j]
    ```

  - **지그재그 순회**

    1행 정방향

    2행 역방향

    3행 정방향

    (...)

    ```python
    # n x m 행렬
    # l : list
    # i : 행
    # j : 열
    for i in range(len(l)):
        for j in range(len(l[0])):
            l[i][j + (m-1-2*j) * (i%2)]
    ```

- **델타**를 이용한 2차 배열 탐색

  2차 배열의 한 좌표에서 4방향(**상하좌우**)의 인접 배열 요소를 탐색하는 방법

  (x, y) 기준 (x `+-` 1, y `+-` 1)의 네 좌표

  - 순서는 본인이 정하기 나름

  ```python
  l = [...]
  delta_x = [-1 , 0, 1, 0]
  delta_y = [0, 1, 0, 1]
  for x in range(len(l)):
      for y in range(len(len(l[x])))
          for i in range(4):
              new_x = x + delta_x[i]
              new_y = y + delta_y[i]
  ```

  - Exercise. delta_sum.py

    abs(이웃 - 자기 자신) 전체의 합 구하기

<br>

<br>

## 행렬곱은 순서에 따라 계산량이 다르다.

- `[2][3] X [3][4] X [4][5]` 세 행렬을 곱할 때,

  1. 앞에서 부터 순서대로

     - `2*3*4 = 24`
     - `2*4*5 = 40`

     총 64 회 계산

  2. 뒤의 행렬 부터

     - `3*4*5 = 60`

     - `2*3*5 = 30`

     총 90 회 계산