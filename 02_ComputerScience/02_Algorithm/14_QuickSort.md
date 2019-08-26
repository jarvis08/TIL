# Quick Sort

## 분할 정복, Divide and Conquer

## 퀵 소트, Quick Sort

### Quick Sort를 사용하는 이유

최악의 경우 O(`n^2`)의 시간 복잡도를 가진다.

하지만 평균적으로 O(`nlog2n`)의 시간 복잡도를 가지며, 가장 많이 사용되는 정렬 방법 중 하나이다.

### 설계 과정

1. Pivot을 값을 정하며, 배열의 중간 값을 자주 사용
2. Pivot을 기준으로 작은 부분과 큰 부분으로 나누어서 다시 Quick Sort를 각각 적용
3. 1~2를 반복

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

  