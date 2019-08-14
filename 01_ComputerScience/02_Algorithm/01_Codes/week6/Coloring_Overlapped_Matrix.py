import sys
sys.stdin = open('overlapped_color.txt', 'r')

T = int(input())
for t in range(1, T+1):
    matrix= []
    for i in range(10):
        matrix += [[0] * 10]
    num_square = int(input())
    info = []
    for i in range(num_square):
        # points, color 저장
        info += [list(map(int, input().split()))]

    # matrix에 color 1 색칠하기
    for i in range(len(info)):
        # color가 1인 경우만
        if info[i][-1] == 1:
            # x좌표 구간
            for x in range(info[i][0], info[i][2]+1):
                # y좌표 구간
                for y in range(info[i][1], info[i][3]+1):
                    # 아직 안칠해진 부분만 색칠
                    if not matrix[x][y]:
                        matrix[x][y] += 1

    # color 2 색칠
    for i in range(len(info)):
        # color가 2인 경우만
        if info[i][-1] == 2:
            # x좌표 구간
            for x in range(info[i][0], info[i][2]+1):
                # y좌표 구간
                for y in range(info[i][1], info[i][3]+1):
                    # color 1이 칠해진 부분만 색칠
                    if matrix[x][y] == 1:
                        matrix[x][y] += 1

    # matrix에서 값이 2인 index count
    cnt = 0
    for i in range(10):
        for j in range(10):
            if matrix[i][j] == 2:
                cnt += 1
    print('#{} {}'.format(t, cnt))
