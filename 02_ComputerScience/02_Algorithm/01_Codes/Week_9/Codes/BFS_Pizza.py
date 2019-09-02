import sys
sys.stdin = open('BFS_Pizza.txt', 'r')


def BFS():
    queue = [0] * N
    for i in range(N):
        queue[i] = [i, cheese[i]]
        cheese[i] = 0

    while len(queue) > 1:
        check = queue.pop(0)
        check[1] = check[1] // 2
        if not check[1]:
            for i in range(M):
                if cheese[i]:
                    check = [i, cheese[i]]
                    cheese[i] = 0
                    break
            else:
                continue
        queue.append(check)
    return queue[0][0]+1

TC = int(input())
for tc in range(1, TC+1):
    N, M = map(int, input().split())
    cheese = list(map(int, input().split()))
    print('#{} {}'.format(tc, BFS()))