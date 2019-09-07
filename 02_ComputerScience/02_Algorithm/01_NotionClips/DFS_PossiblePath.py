import sys
sys.stdin = open('DFS_PossiblePath.txt', 'r')


def AOV(ver):
    for i in range(1, 7):
        # 이웃이면서 방문하지 않은 곳
        if naver[ver][i] and not visited[i]:
            AOV(i)
    visited[ver] = 1

TC = int(input())
for tc in range(1, TC + 1):
    # 노드 개수, 입력 줄 수
    V, E = map(int, input().split())

    naver = [[0]* (V+1) for _ in range(V+1)]
    for _ in range(E):
        start, goal = map(int, input().split())
        naver[start][goal] = 1

    s_ver, t_ver = map(int, input().split())
    visited = [0] * (V+1)

    AOV(s_ver)
    if visited[t_ver]:
        print('result = {}'.format(1))
