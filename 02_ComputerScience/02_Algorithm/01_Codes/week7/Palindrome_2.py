import sys
sys.stdin = open('Palindrome_2.txt', 'r')


def check_pal(row_s, cal_s):
    if row_s==row_s[::-1] or cal_s==cal_s[::-1]:
        return True
    return False


for t in range(1, 11):
    _ = int(input())
    strings = []
    trans_str = [''] * 100
    for i in range(100):
        t_input = input()
        strings += [t_input]
        for j in range(100):
            trans_str[j] += t_input[j]

    for len_pal in range(100, 1, -1):
        max_idx = 100 - len_pal
        found = False
        for i in range(100):
            for j in range(max_idx+1):
                row_str = strings[i][j:j+len_pal]
                col_str = trans_str[i][j:j+len_pal]
                if check_pal(row_str, col_str):
                    print('#{} {}'.format(t, len_pal))
                    found = True
                    break
            if found:
                break
        if found:
            break