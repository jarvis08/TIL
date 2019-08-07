# 정렬, Sort

---

- 정렬 : 2개 이상의 자료를 특정 기준에 의해 Ascending, 혹은 Descending으로 재배열 하는 것

  - `O(n^2)` : 버블 / 삽입 / 선택 정렬
  - `O(nlongn)` : 퀵 / 병합 / 힙 정렬
  - `O(n)` : 카운팅 정렬(특수한 경우에만 사용 가능)

---

## 버블 정렬, Bubble Sort

- 시간 복잡도 = `O(n^2)`

- 인접한 두 개의 원소를 비교하며 자리를 계속 교환하는 방식
- 교환하며 자리를 이동하는 모습이 물 위에 올라오는 거품 모양과 같다고 하여 버블 정렬

- 첫 번째 원소부터 인접한 원소끼리 계속 자리를 교환하면서 맨 마지막 자리까지 이동

- 한 단계가 끝나면 가장 큰 원소가 마지막 자리로 정렬됨

```python
a = [7, 55, 12 , 44, 49]
def BubbleSort(a):
    # 4부터 0까지, 비교할 마지막 원소의 index를 의미
    for i in range(len(a)-1, 0, -1):
        # 0부터 i까지, 인접 원소인 [j], [j+1]을 마지막 원소인 [i]까지 비교
        for j in range(0, i):
            if a[j] > a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
# i 당 5, 4, 3, 2, 1회씩 j를 반복
# n(n+1)/2 = (n^2 + n)/2 => O(n^2)
```

---

## 카운팅 정렬, Counting Sort

- 평균 수행시간 = 최악 수행시간 = `O(N+k)`

  k : 정수의 최대값

- 비교환 방식

  같은 값의 인덱스 순서가 변하지 않음

  e.g., [2, 1, 2] -> [1, 2, 2] 일 때 [2]끼리의 순서가 원본과 같음

- 전제 조건

  1. 공간 할당을 위해 집합 내 가장 큰 값과 최소값을 알 수 있어야하며, 연속된 숫자

     e.g., 0~10 사이의 정수라면 10이라는 값을 알 수 잇다.

  2. type(input) == int 혹은 정수로 표현할 수 있는 자료

     0~1 사이의 확률값 중 소수점 둘째 자리 까지의 숫자 범위는 [0.01, 0.02, ... 0.99]의 값들로 , 100개의 유한한 숫자

- 과정

  1. Data_list, Count_list, Sorted_list 준비

  2. Count_list의 크기는 중복값이 없는 Data_list의 data 개수이며,

     작은 숫자부터 Count_list에 순서대로 배정됨

  3. Count_list에 data 별 개수를 값으로 넣음

  4. Count_list 요소 별로 자신 이전의 누적 개수를 더해줌

  5. Data_list의 끝에서 부터 시작하여,

     Sorted_list[Count_list[data]]에 해당하는 곳에 data를 삽입

  6. Count_list[data]의 값을 -1

---

## 선택 정렬, Selection Sort

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

- 선택 정렬, Selection Sort

- O(N^2)

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

- 달팽이 문제, `SelectionSort_snail.py`

  Tip. 꺾이는 방향의 순서(우, 하, 좌, 상)는 정해져 있으며, 반복된다.

  1. `(i, j)` index에 `direction_delta`를 계속해서 더해준다.
  2. index 범위를 벗어나거나, 값이 이미 있는 경우 방향 전환

  ```python
  direction_delta = [(0, 1), (-1, 0), (0, -1), (1, 0)]
  ```