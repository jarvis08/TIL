def DFS(v):
    visited = [0] * 8
    stack = [0] * 10
    top = -1
    top += 1
    stack[top] = v

    while top != -1:
        # pop 연산
        v = stack[top]
        top -= 1
        if visited[v] != 1:
            visited[v] = 1
            print(v)
            for i in range(1, 8):
                if G[v][i] and not visited[i]:
                    top == 1
                    stack[top] = i

edges = [1, 2, 1, 3, 2, 4, 2, 5, 4, 6 ,7, 6, 6, 7, 3, 7]
G = [[0]*8 for _ in range(8)]

for i in range(0, len(edges), 2):
    G[edges[i]][edges[i+1]] = 1
    G[edges[i+1]][edges[i]] = 1