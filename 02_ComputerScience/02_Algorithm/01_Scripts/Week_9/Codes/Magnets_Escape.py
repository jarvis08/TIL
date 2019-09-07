import sys
sys.stdin = open('Magnets_Escape.txt', 'r')



for tc in range(1, 11):
    input()
    mat = [0] * 100
    for i in range(100):
        mat[i] = list(map(int, input().split()))

    ans = 0
    for x in range(100):
        flag = 0
        for y in range(100):
            if mat[y][x] == 1:
                flag = 1
            elif mat[y][x] == 2 and flag == 1:
                ans += 1
                flag = 0

    print("#%d"%tc, ans)
    print('#{} {}'.format(tc, cnt))