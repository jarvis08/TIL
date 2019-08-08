# Week6 Day3

---

- `SelectSort_snail_review.py`

  - `direction_delta`의 index 간단하게 옮기기

    ```python
    dir_idx = 0
    dir_idx = (dir_stat + 1) % 4
    ```

  - `snail` 행렬의 벽 혹은 값 인지

    ```python
    def isWall(x, y):
        if x < 0 or x >= 5 : return True
        if y < 0 or y >= 5 : return True
        if snail[x][y] != 0: return True
        return False
    ```

- `max_line_review.py`

  - 방법 1, 내가 시도한 방법

    line 별 sum 값을 모두 저장한 후 max를 탐색

  - 방법2

    line별 sum 값을 구할 때 마다 max 값을 갱신

    ```python
    # row
    for i in range(100):
    	for j in range(100):
    		_sum += table[i][j]
    	if _max < _sum: _max = _sum
    
    # column
for i in range(100):
    	for j in range(100):
    		_sum += table[j][i]
    	if _max < _sum: _max = _sum
    ```
    
  - matrix's diagonal summation
  
      ```python
      for i in range(100):
        _sum += table[i][99-i]
      ```