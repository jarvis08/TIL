# SSAFY Week6 Day3

---

## Week5 풀이

- min_max.py

  차례대로 max, min 값과 비교하며 값을 갱신

- electric_bus.py

  1. 이전 위치(`pre`)를 현재 위치(`cur`) 값으로 할당
  2. 전체 `while`문을 통해 현재 위치가 종료 지점 이상의 거리를 이동했는가를 탐색

  2. 현재 위치 변수를 최대 이동거리(`K`) 만큼 이동
  3. `for`문을 이용하여 이동한 지점으로부터 이전 지점과 현재 지점 사이에 충전소가 있는가를 탐색(이동한 지점으로부터 `-1`씩 이동)
  4. 만약(`if`) 충전소를 찾은 지점이 `pre`의 위치와 동일하다면 충전횟수를 `0`으로 하며 `break`

- cards.py

  1. Counting Sort 방식처럼 카드의 숫자들을 인덱스로 하여 count

  2. max index를 `0`으로 설정 후 비교(`<=` 혹은 `>=`이용)해 가며 max index를 구한다.
  3. max index의 index가 개수가 최대인 카드의 숫자

  - 띄워지지 않은 한 자리 숫자의 나열을 input으로 받을 때

    ```python
    cards = input()
    # 숫자 별 개수 구하기
    cnt = [0] * 10
    for i in range(len(cards)):
        cnt[int(cards[i])] += 1
    ```

    ```python
    cards = input()
    card_list = []
    # 숫자 별 개수 구하기
    for i in range(len(cards)):
        card_list.append(int(cards[i]))
    ```

- partial_sum.py

  하나의 요소 별로 part 크기만큼의 횟수로 사용됨

  ```python
  list = [1, 2, 3, 4, 5, 6, 7]
  ```

  - 3개씩의 합들 중 max 값을 구할 때, index 2의 숫자 3은 3회 사용된다.

    (1, 2, `3`) (2, `3`, 4) (`3`, 4, 5)

    이를 생각해 봤을 때, 세번씩 계속 더하는건 효율 적이지 않다.

    Do not recompute!

  - 해결 방법 2가지

    - 방법 1

      1. (1, 2, 3) = 6
      2. (2, 3, 4) = 6 - 1 + 4 = 9
      3. (3, 4, 5) = 8 - 2 + 5 = 11

    - 방법 2, Sliding Window

      (4, 5, 6)까지의 합을 구하려면 [(1~6) - (1~3)]

- flatten.py

  - 큰값 - 1, 작은 값 + 1로 인해 data의 status가 계속 변경된다.

  - dump 횟수가 충분히 크며 전체 개수가 column 수에 비례하다면, `전체 개수 / column 개수`

  - max, min을 구했을 때 그 둘의 차이가 1 혹은 0 이면 종료

  - 상위 level 방법

    Counting Sort를 응용하여 count 기반의 dump

---

## 2차원 배열

- 1차원 list를 묶어놓은 list

- 2차원 이상의 다차원 list는 차원에 따라 index를 선언

- 3차원 list의 선언 : 세로 길이(행 개수), 가로 길이(열 개수)를 필요로 함

- Python에서는 데이터 초기화를 통해 변수선언과 초기화가 가능

- 배열 순회

  n x m 배열의 n*m 개의 모든 원소를 빠짐 없이 조사하는 방법

  - 행 우선 순회

    ```python
    # l : list
    # i : 행
    # j : 열
    for i in range(len(l)):
        for j in range(len(l[i])):
            l[i][j]
    ```

  - 열 우선 순회

    ```python
    # l : list
    # i : 행
    # j : 열
    for j in range(len(l[0])):
        for i in range(len(l)):
            l[i][j]
    ```

  - 지그재그 순회

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

- 델타를 이용한 2차 배열 탐색

  2차 배열의 한 좌표에서 4방향(상하좌우)의 인접 배열 요소를 탐색하는 방법

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

---

## 전치 행렬

- 가로 방향에 대해, 그리고 세로 방향에 대해 동일한 문제를 해결할 때 사용

  가로 방향에 대해 문제를 해결한 후, 전치하여 다시 적용

```python
# Original
1 2 3
4 5 6
7 8 9
```

```python
# Transpose
1 4 7
2 5 8
3 6 9
```

```python
l = [[], [], []]
for i in range(3):
    for j in range(3):
        if i < j :
            l[i][j], l[j][i]= l[j][i], l[i][j]
```

## 부분 집합(Subset)의 합

- 유한 개의 정수로 이루어진 집합이 존재할 때,

  부분집합 중에서 그 집합의 원소를 모두 더한 값이 0이 되는 경우가 있는가?

1. 완전 검색 기법

      집합의 모든 부분집합을 생성한 후에 각 부분집합의 합을 계산

        - 파스칼의 삼각형 응용 (재귀 함수 이용 가능)

          {1, 2, 3} 집합이 있으며, 각 원소의 등장 여부를 1과 0으로 표시

          - 111, 110, 101, 100, 011, 010, 001, 000

            = {1, 2, 3}, {1, 2}, {1, 3}, {1}, {2, 3}, {2}, {3}, {none}

          ```python
          a = [0, 0, 0]
          for i in rnage(2):
              a[0] = i
               for j in range(2):
                      a[1] = j
                      for k in range(2):
                          a[2] = k
          				print(a)
          ```

      - 집합의 원소 개수와 중첩된 for문 개수가 비례하며,

        경우의 수는 `2 ** (원소개수)`

2. 비트(Binary Digit) 연산 이용

      - 비트 연산 기초

        0과 1로만 구성

        - `&`, AND

          `1, True` = (1, 1)

        - `|`, OR

          `1, True` = (1, 1), (1, 0), (0, 1)

        - `^`, exclusive, XOR

          `True` = (1, 0), (0, 1)

        - `<<`

          `x = 1` 일 때 `x << n` = `2 ** n`

          ```
          x = 1 일 때
          x의 2진법 = 001
          x << 1 :: 010 = 2
          x << 2 :: 100 = 4
          ```

        - `>>`

          ```
          x = 12 일 때,
          x는 2진법으로 1100
          x >> 1 :: 0110 = 6
          x >> 2 :: 0011 = 3 
          ```

        - `i & (1 << j)`

          `i` : 어떤 수

          `j` : `j` 번째 비트

          - `1 << j`는 j 비트를 제외하고는 모두 masking

          - `&`로 인해 `i`의 `j`번째 비트가 1이면 `True`, 0이면 `False`

            i.e., i의 j번째 비트가 무엇인지 알고 싶을 때 사용

      - 1(True), 0(False) 두 값만 필요하므로 Binary Counting을 이용하여 풀 수 있다.

        ```
        111 = 7
        110 = 6
        101 = 5
        100 = 4
        011 = 3
        010 = 2
        001 = 1
        000 = 0
        ```

        ```python
        # 부분집합 출력해보기
        arr = [3, 6, 7, 1, 5, 4]
        # 원소 개수
        n = len(arr)
        
        # 1 << n : 부분집합의 개수 i
        for i in range(1 << n):
            # 원소의 개수(j = 0 ~ 5)만큼 비트를 비교
            for j in range(n):
                # i의 j번째 비트가 1이면 j번째 원소를 출력
                if i & (1 << j):
                    print(arr[j], end=', ')
            print()
        ```

        - `i == 47` 일 때,

          47 = 32 + 8 + 4  + 2 + 1 = 101111

          - `j == 0`

            `101111`와 `000001`을 `&` = 1

            `101111`와 `000010`을 `&` = 1

            `101111`와 `000100`을 `&` = 1

            `101111`와 `001000`을 `&` = 1

            `101111`와 `010000`을 `&` = 0

            `101111`와 `100000`을 `&` = 1

          - i.e., 48번째(`i=47`) subset은 `i & (1 << j) = 1`을 만족하는 `arr[j]`들의 집합인 {3, 6, 7, 1, 4}

3. Exercise. 정수 집합 중, 부분 집합의 합이 0이되는 부분 집합 구하기

      ```python
      origin_set = [1, 2, -1, -3, 4, -2, 5]
      n = len(origin_set)
      subsets = []
      for i in range(1 << n):
          subset = []
          for j in range(n):
              if i & (1 << j):
                  subset += [origin_set[j]]
          subsets += [subset]
      
      sum_zero = []
      for sub in subsets:
          sum_sub = 0
          if not len(sub):
              # 공집합 제거
              continue
          for element in sub:
              sum_sub += element
          if not sum_sub:
              sum_zero += [sub]
      print(sum_zero)
      ```

---

## 검색, Search

- 저장되어 있는 자료 중에서 원하는 항목을 찾는 작업

- 탐색 공간에서 목적하는 탐색 키를 가진 항목을 찾는 것

  탐색 키(search key) : 자료를 구별하여 인식할 수 있는 키

- 검색의 종류

  - 순차 검색(sequential search), O(n)

    index 0부터 비교

    1. 탐색 공간이 정렬되어있지 않은 경우

        ```python
        def sequentialSearch(a, n, key):
            i = 0
            while i < n and a[i]!=key:
                i += 1
            if i < n:
                return i
            # 못 찾았을 경우
            return -1
        ```

    2. 탐색 공간이 정렬되어 있을 경우

       비교 도중 data의 값이 탐색 키 보다 크다면, 나머지 탐색 공간은 탐색 필요가 없다.

  - **이진 검색(binary search)**

    - 효율성이 매우 높은 검색 알고리즘

    - 보간 검색의 임의 설정 방법이 'index / 2'로 정해져 있는 검색 방법

      탐색 공간을 계속해서 1/2로 감소시킨다.

      - 보간 검색

        적당한 값의 index를 임의로 설정한 후 해당 값을 비교하여 점차 범위를 좁혀 나가는 과정

    - 코드 구현, BinarySearch.py

        - 검색 범위의 시작점과 종료점을 이용하여 검색을 반복 수행
        - 이진 검색을 사용하는 module의 경우, dataset에 삽입이나 삭제가 발생했을 때 배열의 상태를 항상 정렬 상태로 유지하는 추가 작업이 필요
        
        ```python
        def binarySearch(a, key):
            start = 0
            end = len(a) - 1
            while start <= end:
                middle = (start + end) // 2
                print("start : ", start)
                print("end : ", end)
                print("middle : ", middle)
                if a[middle] == key:
                    return True
                elif a[middle] > key:
                    end = middle - 1
                else:
                    start = middle + 1
                print('---------------')
            return False
        
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 21, 23]
        print(binarySearch(a, 20))
        ```
        
    - 재귀 함수로 구현하기, BinarySearch_Recursive.py
  
        ```python
        def binarySearch2(a, low, high, key):
            if low > high:
                return False
            else:
                middle = (low + high) // 2
                if key == a[middle]:
                    return True
                elif key < a[middle]:
                    return binarySearch2(a, low, middle - 1, key)
                elif a[middle] < key:
                    return binarySearch2(a, middle + 1, high, key)
        ```
  
  - 해쉬(hash)
  
    - Index
  
      - DB에서 유래했으며, 테이블에 대한 동작 속도를 높여주는 자료 구조
  
      - 인덱스의 저장 공간은 테이블을 저장하는데 필요한 공간보다 작다.
  
        인덱스는 key-field 만을 갖고 있으며, 테이블은 다른 세부 항목을 보유하기 때문이다.
  
      - 배열을 사용한 인덱스
  
        대량의 데이터를 매번 정렬하면 속도 저하 불가피
  
        배열 인덱스를 이용하여 문제 해결

---

## 정렬

- 선택 알고리즘, Selection Algorithm, `O(KN)`

  저장되어 있는 data로부터 k번째로 큰/작은 원소를 찾는 방법

  (= 최대/최대/중간값을 찾는 알고리즘)

  1. 정렬 알고리즘을 이용하여 data 정렬
  2. 원하는 순서에 있는 원소 가져오기

  - `K = N` 일 경우 **선택 정렬**과 동일

  ```python
  def select(l, k):
      for i in range(k):
          min_idx = i
          for j in range(i+1, len(l)):
              if l[min_idx] > l[j]:
                  min_idx = j
          l[i], l[min_idx] = l[min_idx], l[i]
      return l[k-1]
  ```

- 선택 정렬, Selection Sort, `O(N**2)`

  주어진 data 중 가장 작은 값의 원소부터 차례대로 선택하여 위치를 교환

  (계속해서 가장 큰/작은 값을 앞 인덱스로 위치시키고, 나머지 부분에 대해 작업을 반복)

  1. 주어진 list 중 최소값 찾기
  2. 그 값을 맨 앞에 위치한 값과 교환
  3. 맨 처음 위치를 제외한 나머지 리스트를 대상으로 위의 과정을 반복

  ```python
  def selectionSort(l):
      for i in range(len(l)-1):
          min_idx = i
          for j in range(i+1, len(l)):
              if l[min_idx] > l[j]:
                  min_idx = j
          l[i], l[min_idx] = l[min_idx], l[i]
      return l
  ```

- 달팽이 문제, Selection_snail.py

  Tip. 꺾이는 방향의 순서(우, 하, 좌, 상)는 정해져 있으며, 반복된다.

  1. `(i, j)` index에 `direction_delta`를 계속해서 더해준다.
  2. index 범위를 벗어나거나, 값이 이미 있는 경우 방향 전환

  ```python
  direction_delta = [(0, 1), (-1, 0), (0, -1), (1, 0)]
  ```

  