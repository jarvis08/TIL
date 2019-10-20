import sys
sys.stdin = open('UN_Storage.txt', 'r')


def isWall(val):
    if val == N:
        return True
    return False


def getRowEnd(x, j_1, j_2):
    max_i = x
    for i in range(x, N):
        if not mat[i][j_1]:
            break
        for j in range(j_1, j_2+1):
            mat[i][j] = 0
        max_i = i
    return max_i


TC = int(input())
for tc in range(1, TC+1):
    N = int(input())
    mat = []
    for _ in range(N):
        mat.append(list(map(int, input().split())))

    bombs_i = []
    bombs_j = []
    for i in range(N):
        found = False
        for j in range(N):
            if found and (isWall(j+1) or not mat[i][j+1]):
                bombs_j.append(j)
                found = False
                bombs_i.append(getRowEnd(i, bombs_j[-2], bombs_j[-1]))
                continue
            
            if found:
                continue

            if mat[i][j]:
                found = True
                bombs_i.append(i)
                bombs_j.append(j)
            
            if found and (isWall(j+1) or not mat[i][j+1]):
                bombs_j.append(j)
                found = False
                bombs_i.append(getRowEnd(i, bombs_j[-2], bombs_j[-1]))

    mn_form = []
    while bombs_i:
        m = bombs_i.pop() + 1
        m -= bombs_i.pop()
        n = bombs_j.pop() + 1
        n -= bombs_j.pop()
        mn_form.extend([[m, n, m*n]])
    
    for i in range(len(mn_form)):
        max_val = 0
        max_idx = 0
        found = False
        for j in range(len(mn_form)-i):
            if mn_form[j][2] > max_val:
                max_val = mn_form[j][2]
                max_idx = j
                found = True
                continue
            elif mn_form[j][2] == max_val:
                if mn_form[j][0] > mn_form[max_idx][0]:
                    max_val = mn_form[j][2]
                    max_idx = j
                    found = True
                    continue

            if found and j == len(mn_form)-i-1:
                mn_form[max_idx], mn_form[len(mn_form)-i-1] = mn_form[len(mn_form)-i-1], mn_form[max_idx]
        

    result = []
    for i in range(len(mn_form)):
        result.append(str(mn_form[i][0]))
        result.append(str(mn_form[i][1]))
    print('#{} {} {}'.format(tc, len(mn_form), ' '.join(result)))
