import sys
sys.stdin = open('GNS-Sort_Stringed_Int_input.txt', 'r')


T = int(input())
for tc in range(1, T+1):
    input()
    s_in = input().split()
    s_list = ["ZRO", "ONE", "TWO", "THR", "FOR", "FIV", "SIX", "SVN", "EGT", "NIN"]

    c_list = [0] * 10
    for s in s_in:
        for i in range(10):
            if s == s_list[i]:
                c_list[i] += 1
    
    result = ''
    for i in range(10):
        result += (s_list[i] + ' ') * c_list[i] 
    print('#{}\n{}'.format(tc, result))