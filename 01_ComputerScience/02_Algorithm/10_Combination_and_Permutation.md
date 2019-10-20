# 순열과 조합

## 조합, Combination

중복이 불가하며, 순서가 상관이 없을 때

`n`C`r` = `n!` / {`r! * (n-r)!`}

```python
from itertools import combinations
print(list(combinations('빨주노초',2)))
"""
[('빨', '주'), ('빨', '노'), ('빨', '초'), ('주', '노'), ('주', '초'), ('노', '초')]"""
```

<br>

### 중복 조합, Combination with Repetition

중복이 가능하며, 순서가 상관이 없을 때

`n`H`r` = `n+r-1`C`r`

```python
from itertools import combinations_with_replacement
print(list(combinations_with_replacement("빨주노초",2)))
"""
[('빨', '빨'), ('빨', '주'), ('빨', '노'), ('빨', '초'), ('주', '주'), ('주', '노'), ('주', '초'), ('노', '노'), ('노', '초'), ('초', '초')]"""
```

<br><br>

## 순열, Permutation

중복이 불가하며, 순서가 상관이 있을 때

`n`P`r` = `n * r`

```python
from itertools import permutations
per = permutations(["빨","주","노","초"],2)
print(list(per))
"""
[('빨', '주'), ('빨', '노'), ('빨', '초'), ('주', '빨'), ('주', '노'), ('주', '초'), ('노', '빨'), ('노', '주'), ('노', '초'), ('초', '빨'), ('초', '주'), ('초', '노')]"""
```

<br>

### 중복 순열, Permutation with Repetition

중복이 가능하며, 순서가 상관이 있을 때

`n`Π`r` = `n^r`

i.g., 1~3의 숫자가 적힌 카드덱이 두 개 있을 때, 각 덱에서 하나씩 뽑는 경우의 수

[1,1], [1, 2], [1,3], ..., [3, 1], [3, 2], [3, 3] 총 9가지

```python
from itertools import product
per1 = product((["빨","주","노","초"]), repeat=2)
print(list(per1))
"""
[('빨', '빨'), ('빨', '주'), ('빨', '노'), ('빨', '초'), ('주', '빨'), ('주', '주'), ('주', '노'), ('주', '초'), ('노', '빨'), ('노', '주'), ('노', '노'), ('노', '초'), ('초', '빨'), ('초', '주'), ('초', '노'), ('초', '초')]"""
```