import sys
sys.stdin = open('Samsung_Bus.txt', 'r')


TC = int(input())
for tc in range(1, TC+1):
    N = int(input())
    go = []
    for _ in range(N):
        n = map(int, input().split())
        go.append(tuple(n))

    P = int(input())
    stops = []
    for _ in range(P):
        stops.append([int(input()), 0])

    for n in range(N):
        for i in range(go[n][0], go[n][1] + 1):
            for j in range(P):
                if stops[j][0] == i:
                    stops[j][1] += 1

    result = []
    for p in range(P):
        result.append(str(stops[p][1]))
    print('#{} {}'.format(tc, ' '.join(result)))
