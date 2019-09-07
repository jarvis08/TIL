import sys
sys.stdin = open('Sum_Subsets.txt', 'r')


def sum_subset(subset):
    _sum = 0
    for num in subset:
        _sum += num
    return _sum


T = int(input())
A = []
for i in range(1, 13):
    A += [i]

for t in range(1, T+1):
    N, K = tuple(map(int, input().split()))
    cnt = 0
    for i in range(1 << 12):
        subset = []
        for j in range(12):
            if i & (1 << j):
                subset += [A[j]]
        if len(subset) == N and sum_subset(subset) == K:
            cnt += 1
    print('#{} {}'.format(t, cnt))