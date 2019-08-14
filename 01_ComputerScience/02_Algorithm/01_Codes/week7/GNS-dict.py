import sys
sys.stdin = open('GNS-Sort_Stringed_Int_input.txt', 'r')

T = int(input())
for tc in range(1, T+1):
    n_str = int(input().split()[1])
    stringed_int = list(input().split())
    s_list = ["ZRO", "ONE", "TWO", "THR", "FOR", "FIV", "SIX", "SVN", "EGT", "NIN"]
    
    # string 별 개수
    c_dict = dict()
    for s in stringed_int:
        if s not in c_dict.keys():
            c_dict[s] = 1
        elif s in c_dict.keys():
            c_dict[s] += 1
    
    # 순서 고려를 위해 list를 활용
    f_list = []
    for s in s_list:
        if s in c_dict.keys():
            f_list += [s]

    print('#{}'.format(tc))
    for k in f_list:
        for _ in range(c_dict[k]):
            print(k, end=' ')
    print()