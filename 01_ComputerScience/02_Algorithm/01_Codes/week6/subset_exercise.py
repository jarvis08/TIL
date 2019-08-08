# 정수 집합 중, 부분집합의 합이 0이되는 것이 존재하는지 확인하는 함수
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