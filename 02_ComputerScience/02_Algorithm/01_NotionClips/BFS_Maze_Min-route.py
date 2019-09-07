import sys
sys.stdin = open('BFS_Maze_Min-route.txt', 'r')


def BFS(i, j):
    queue = [[i, j, 0]]
    visited[i][j] = 1
    while queue:
        now = queue.pop(0)
        x = now[0]
        y = now[1]
        k = now[2]
        
        for d in range(4):
            m = x + dx[d]
            n = y + dy[d]
            if m < 0 or n < 0 or m >= N or n >= N:
                continue
            if maze[m][n] == '1':
                continue
            if visited[m][n] == 1:
                continue

            if maze[m][n] == '3':
                return k
            queue.append([m, n, k+1])
            visited[m][n] = 1
    return 0


# 상 하 좌 우
dx = [1, -1, 0, 0]
dy = [0, 0, -1, 1]

TC = int(input())
for tc in range(1, TC+1):
    N = int(input())
    maze = []
    for i in range(N):
        maze += [input()]

    visited = [[0] * N for _ in range(N)]
    found = False
    for i in range(N):
        if found:
            break
        for j in range(N):
            if maze[i][j] == '2':
                print('#{} {}'.format(tc, BFS(i, j)))
                found = True
                break
    
    