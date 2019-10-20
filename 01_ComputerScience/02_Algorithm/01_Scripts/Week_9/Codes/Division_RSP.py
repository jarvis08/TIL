import sys
sys.stdin = open('Division_RSP.txt', 'r')


def rsp(value, idx):
    if value[0]==1 and value[1]==3:
        return [value[0], idx[0]]
    elif value[0]==2 and value[1]== 1:
        return [value[0], idx[0]]
    elif value[0]==3 and value[1]==2:
        return [value[0], idx[0]]
    elif value[0]==value[1]:
        return [value[0], idx[0]]
    else:
        return [value[1], idx[1]]


def div_rsp(value, idx):
    half = int(len(value)//2)
    
    L = value[:half]
    L_idx = idx[:half]

    R = value[half:]
    R_idx = idx[half:]
    
    match_1 = []
    match_2 = []

    # 왼쪽
    if len(L) == 2:
        match_1 = rsp(L, L_idx)
    elif len(L) == 1:
        match_1 = [L[0], L_idx[0]]
    else:
        match_1 = div_rsp(L, L_idx)
    
    # 오른쪽
    if len(R) == 2:
        match_2 = rsp(R, R_idx)
    elif len(R) == 1:
        match_2 = [R[0], R_idx[0]]
    else:
        match_2 = div_rsp(R, R_idx)

    # semi-final
    sf_val = [match_1[0], match_2[0]]
    sf_idx = [match_1[1], match_2[1]]
    return rsp(sf_val, sf_idx)


TC = int(input())
for tc in range(1, TC+1):
    N = int(input())
    RSP = list(map(int, input().split()))
    idx = [i for i in range(1, N+1)]
    print('#{} {}'.format(tc, div_rsp(RSP, idx)[1]))
