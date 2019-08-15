# 전치 행렬, Transpose

---

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