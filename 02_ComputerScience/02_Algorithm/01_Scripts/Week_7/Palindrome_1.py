import sys
sys.stdin = open('Palindrome_1.txt', 'r')


def check_pal(string):
    if string == string[::-1]:
        return True
    return False


for t in range(1, 11):
    len_pal = int(input())
    # 0 ~ (8-len_pal) idx로 시작해야 len_pal만큼의 길이 확보가 가능
    max_idx = 8 - len_pal

    # 2-D list에 matrix 저장
    strings = []
    for i in range(8):
        t_input = input()
        t_list = []
        for a in t_input:
            t_list += [a]
        strings += [t_list]
    
    cnt = 0
    # check strings
    for i in range(8):
        for j in range(max_idx+1):
            row_str = ''
            col_str = ''
            for k in range(len_pal):
                row_str += strings[i][j+k]
                col_str += strings[j+k][i]
            if check_pal(row_str):
                cnt +=1
            if check_pal(col_str):
                cnt +=1

    print('#{} {}'.format(t, cnt))
    

    

