#-*- coding:utf-8 -*-
arr = [3, 6, 7, 1, 5, 4]
# 원소 개수
n = len(arr)

# 1 << n : 부분집합의 개수
for i in range(1 << n):
    # 원소의 개수만큼 비트를 비교
    for j in range(n):
        # i의 j번째 비트가 1이면 j번째 원소를 출력
        if i & (1 << j):
            print(arr[j], end=', ')
    print()