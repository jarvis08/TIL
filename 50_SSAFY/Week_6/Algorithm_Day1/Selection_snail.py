original = [
    [9, 20, 2, 18, 11],
    [19, 1, 25, 3, 21],
    [8, 24, 10, 17, 7],
    [15, 4, 16, 5, 6],
    [12, 13, 22, 23, 14]
]
"""Tip
꺾이는 방향의 순서(우, 하, 좌, 상)는 정해져 있으며, 반복된다.
1. (i, j) index에 direction_delta를 계속해서 더해준다.
2. index 범위를 벗어나거나, 값이 이미 있는 경우 방향 전환
"""
direction = [[0, 1], [1, 0], [0, -1], [-1, 0]]

snail = []
temp = []
for i in range(1, 26):
    temp += [0]
    if i % 5 == 0:
        snail += [temp]
        temp = []

d = 0
x = 0
y = 0

for k in range(25):
    min_i = 0
    min_j = 0
    min_val = original[0][0]
    # 최소값의 인덱스와 값 찾기
    for i in range(5):
        for j in range(5):
            if min_val > original[i][j]:
                min_i = i
                min_j = j
                min_val = original[i][j]
    # 최소값을 충분히 큰 수로 설정하여, 다음 최소값 찾기에서 제외
    original[min_i][min_j] = 100

    # 달팽이
    # 값 넣기
    snail[x][y] = min_val
    # 종료 조건 확인 및 이동
    # ('벽' or '0 아닌 값')이면 방향 전환
    test_x = x + direction[d][0]
    test_y = y + direction[d][1]
    if test_x > 4 or test_y > 4 or test_x < 0 or test_y < 0 or snail[test_x][test_y] != 0:
        if d == 3:
            d = 0
        else:
            d += 1
    x += direction[d][0]
    y += direction[d][1]
print(snail)

