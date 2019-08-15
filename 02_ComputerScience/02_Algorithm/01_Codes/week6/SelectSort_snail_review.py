original = [
    [9, 20, 2, 18, 11],
    [19, 1, 25, 3, 21],
    [8, 24, 10, 17, 7],
    [15, 4, 16, 5, 6],
    [12, 13, 22, 23, 14]
]

snail = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

X = 0
Y = 0

# 우, 하, 좌, 상
dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]
dir_stat = 0

def isWall(x, y):
    if x < 0 or x >= 5 : return True
    if y < 0 or y >= 5 : return True
    if snail[x][y] != 0: return True
    return False

def sel_min():
    return 0

# 현재 turn의 min값
cur_min = 0

for i in range(25):
    # sel_min() : select sort로 찾은 min값 반환
    cur_min = sel_min()
    # 달팽이에 최소값 삽입
    snail[X][Y] = cur_min
    X += dx[dir_stat]
    X += dx[dir_stat]

# 벽 혹은 값이 있는지 확인
if isWall(X, Y):
    X -= dx[dir_stat]
    Y -= dy[dir_stat]
    # direction delta 변경
    dir_stat = (dir_stat + 1) % 4
    X += dx[dir_stat]
    X += dx[dir_stat]

for i in range(5):
    for j in range(5):
        print(snail[i][j], end='')
    print()