import sys
sys.stdin = open('BFS_Min_Path.txt', 'r')


def BFS(start, target):
    queue = [(start, 0)]
    visited[start] = 1
    while queue:
        ver, k = queue.pop(0)
        for i in range(1, V+1):
            if connected[ver][target]:
                return k + 1
            if visited[i]:
                continue
            if connected[ver][i]:
                queue.append((i, k+1))
                visited[i] = 1

TC = int(input())
for tc in range(1, TC+1):
    # V : number of vertex
    # E : number of connectedions
    V, E = map(int, input().split())
    
    visited = [0]*(V+1)
    start = 0
    target = 0
    connected = [[0]*(V+1) for _ in range(V+1)]
    for i in range(E):
        node, to = map(int, input().split())
        connected[node][to] = 1
        connected[to][node] = 1
    S, G = map(int, input().split())
    cnt = BFS(S, G)
    print('#{} {}'.format(tc, cnt))