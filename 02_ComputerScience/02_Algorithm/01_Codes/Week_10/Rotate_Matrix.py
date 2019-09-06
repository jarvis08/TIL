import sys
sys.stdin = open('Rotate_Matrix.txt', 'r')

import copy

TC = int(input())
for tc in range(1, TC+1):
    N = int(input())
    matrix = []
    for _ in range(N):
        matrix.append(list(input().split()))

    # 90, 180, 270 시계방향
    moved = []
    for _ in range(N):
        moved.append([0]*N)

    result = []
    print('#'+str(tc))
    for r in range(3):
        for i in range(N):
            for j in range(N):
                moved[j][N-i-1] = matrix[i][j]
        matrix = copy.deepcopy(moved)
        if r == 0:
            for i in range(N):
                result.append([''.join(matrix[i])])
        else:
            for i in range(N):
                result[i].append(''.join(matrix[i]))
    for i in range(N):
        print(' '.join(result[i]))