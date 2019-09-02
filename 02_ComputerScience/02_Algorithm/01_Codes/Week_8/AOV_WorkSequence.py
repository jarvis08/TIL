import sys
sys.stdin = open('AOV_WorkSequence.txt', 'r')


def reverse_DFS(v):
    for i in range(n_ver+1):
        if naver[v][i] and not visited[i]:
            reverse_DFS(i)
    visited[v] = True
    print(v, end=' ')

for tc in range(1,11):
    n_ver, n_act = map(int, input().split())
    vertex = list(map(int, input().split()))

    naver = [[0]*(n_ver+1) for _ in range(n_ver+1)]
    visited = [0] * (n_ver+1)

    for i in range(n_act):
        naver[vertex[2*i+1]][vertex[2*i]] = 1
    

    print('#{} '.format(tc), end='')
    for i in range(1, n_ver+1):
        if not visited[i]:
            reverse_DFS(i)
    print()