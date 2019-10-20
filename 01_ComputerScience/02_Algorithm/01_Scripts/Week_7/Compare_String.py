import sys
sys.stdin = open('Compare_String.txt', 'r')


# Boyer-Moore Algorithm
def next_idx(cur_i, pattern, string):
    len_p = len(pattern)
    len_str = len(string)
    next_i = 0

    # 패턴이 있는지
    for i in range(len_p):
        if pattern[i] != string[cur_i+i]:
            break
        if i == len_p-1 and pattern[i] == string[cur_i+i]:
            return 1

    # 패턴의 앞글자와 동일한게 있는지
    for i in range(1, len_p):
        if pattern[0] == string[cur_i + i]:
            return next_idx(next_i+i, pattern, string)

    # 패턴 길이 만큼 더한 값을 다음 인덱스로 지정
    next_i = cur_i + len_p
    # (총 길이 - 다음 인덱스)가 패턴 길이보다 길다면 종료
    if next_i+len_p > len_str:
        return 0
    return next_idx(next_i, pattern, string)


TC = int(input())
for t in range(1, TC+1):
    pattern = input()
    string = input()
    result = next_idx(0, pattern, string)
    print('#{} {}'.format(t, result))
    