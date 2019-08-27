# Divide and Conquer, Quick Sort

## 분할 정복, Divide and Conquer

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

## 퀵 소트, Quick Sort

### Quick Sort

  주어진 배열을 두 개로 분할하고, 각각을 정렬

  - 최악의 경우 O(n^2)의 시간복잡도를 가지지만,

    평균적으로는 O(nlogn)이기 때문에 가장 많이 사용됨

  - 합병 정렬과의 차이

    1. 합병 정렬은 그냥 두 부분으로 나누는 반면,

       퀵 정렬은 분할 할 때 기준 아이템(pivot item)을 중심으로 pivot 보다 작은 것은 왼편, 큰 것은 오른편에 위치시키는 규칙이 존재

    2. 각 부분의 정렬이 끝난 후 합병 정렬은 '합병'이라는 후처리 작업이 필요

       퀵정렬은 아무런 작업이 필요 없다.

### 설계 과정

- 방법 1

    1. Pivot을 값을 정하며, 배열의 중간 값을 자주 사용
    2. Pivot을 기준으로 작은 부분과 큰 부분으로 나누어서 다시 Quick Sort를 각각 적용
    3. 1~2를 반복
    
- 방법 2

    1. pivot 값을 기준으로,

       - 좌측에서는 pivot 보다 큰 값을 탐색
       - 우측에서는 pivot 보다 작은 값을 탐색

    2. 구해 진 큰 값과 작은 값의 위치를 교환

    3. 만약 L<R이 성립되는 조건아래 교환할 값들을 찾지 못한다면,

       L/R이 위치한 값과 Pivot의 위치를 바꿈

### 구현

- ratsgo's blog Code

  ```python
  def quick_sort(array):
      len_array = len(array)
      if len_array <= 1:
          return array
      else:
          pivot = array[0]
          greater = [ element for element in array[1:] if element > pivot ]
          lower = [ element for element in array[1:] if element <= pivot ]
          return quick_sort(lower) + [pivot] + quick_sort(greater)
  ```

- [DaleSeo](https://www.daleseo.com/sort-quick/) Code

  ```python
  def quick_sort(arr):
      if len(arr) <= 1:
          return arr
      pivot = arr[len(arr) // 2]
      lower, equal, greater = [], [], []
      for num in arr:
          if num < pivot:
              lower.append(num)
          elif num > pivot:
              greater.append(num)
          else:
              equal.append(num)
      return quick_sort(lesser) + equal + quick_sort(greater)
  ```

- 방법 2

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