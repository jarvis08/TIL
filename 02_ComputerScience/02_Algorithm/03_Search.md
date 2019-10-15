# 검색, Search

- 저장되어 있는 자료 중에서 원하는 항목을 찾는 작업

- 탐색 공간에서 목적하는 탐색 키를 가진 항목을 찾는 것

  탐색 키(search key) : 자료를 구별하여 인식할 수 있는 키

<br>

<br>

## 순차 검색, Sequential Search

- O(N)

  index 0부터 끝까지 비교하여 일치하는 index를 탐색

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

<br>

<br>

## 이진 검색, Binary Search

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

<br>

<br>

## 해쉬, Hash

- DB에서 유래했으며, 테이블에 대한 동작 속도를 높여주는 자료 구조

- 인덱스의 저장 공간은 테이블을 저장하는데 필요한 공간보다 작다.

  인덱스는 key-field 만을 갖고 있으며, 테이블은 다른 세부 항목을 보유하기 때문이다.

- 배열을 사용한 인덱스

  대량의 데이터를 매번 정렬하면 속도 저하 불가피

  배열 인덱스를 이용하여 문제 해결