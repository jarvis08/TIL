import sys
sys.stdin = open('Palindrome_3.txt', 'r')


def check_pal(string):
    if string == string[::-1]:
        return True
    return False


TC = int(input())
for t in range(1, TC+1):
    len_row, len_pal = map(int, input().split())
    max_idx = len_row - len_pal

    strings = []
    trans_str = [''] * len_row
    for i in range(len_row):
        t_input = input()
        strings += [t_input]
        for j in range(len_row):
            trans_str[j] += t_input[j]
    
    found = False
    for i in range(len_row):
        for j in range(max_idx+1):
            row_str = strings[i][j:j+len_pal]
            col_str = trans_str[i][j:j+len_pal]
            if check_pal(row_str):
                print('#{} {}'.format(t, row_str))
                found = True
                break
            if check_pal(col_str):
                print('#{} {}'.format(t, col_str))
                found = True
                break
        if found:
            break