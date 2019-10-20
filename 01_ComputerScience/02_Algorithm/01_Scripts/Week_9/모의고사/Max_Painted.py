import sys
sys.stdin = open('Max_Painted.txt', 'r')


TC = int(input())
for tc in range(1, TC+1):
    N, M, K = map(int, input().split())

    p = []
    for k in range(K):
        p.extend([list(map(int, input().split()))])

    matrix = [[0]*M for _ in range(N)]
    for k in range(K):
        paint = True
        for i in range(p[k][0], p[k][2]+1):
            for j in range(p[k][1], p[k][3]+1):
                if matrix[i][j] > p[k][4]:
                    paint = False
                    break
            if not paint:
                break
        if not paint:
            continue
        else:
            for i in range(p[k][0], p[k][2] + 1):
                for j in range(p[k][1], p[k][3] + 1):
                    matrix[i][j] = p[k][4]

    cnt = [0]*11
    for i in range(N):
        for j in range(M):
            cnt[matrix[i][j]] += 1
    print('#{} {}'.format(tc, max(cnt)))