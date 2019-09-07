import sys
sys.stdin = open('Escape_Maze.txt', 'r')


def backtrack(i, j):
    visited[i][j] = 1
    for d in range(4):
        x = i + direction[d][0]
        y = j + direction[d][1]
        if x < 0 or y < 0 or x >= N or y >= N:
            continue
        if visited[x][y]:
            continue

        # Solution Check
        if maze[x][y] == '3':
            return 1
        
        # 0인 곳을 탐색
        if maze[x][y] == '0':
            check = backtrack(x, y)
            if check:
                return 1
    return 0


# 상하좌우
direction = [(-1, 0), (1, 0), (0, -1), (0, 1)]

TC = int(input())
for tc in range(1, TC+1):
    N = int(input())
    visited = [[0]*N for _ in range(N)]
    found = False

    maze = []
    for i in range(N):
        maze += input().split()
    
    find_start = False
    for i in range(N):
        if find_start:
            break
        for j in range(N):
            if maze[i][j] == '2':
                find_start = True
                result = backtrack(i, j)
                print('#{} {}'.format(tc, result))
                break
