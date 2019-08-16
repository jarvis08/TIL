import sys
sys.stdin = open('Find_Max_Used.txt', 'r')


def max_used(pattern, string):
    done = []
    max_cnt = 0
    for p in pattern:
        if p in done:
            continue
        cnt = 0
        for s in string:
            if p == s:
                cnt += 1
        done += [p]
        if cnt > max_cnt:
            max_cnt = cnt
    return max_cnt

TC = int(input())
for tc in range(1, TC+1):
    p = input()
    s = input()
    max_cnt = max_used(p, s)
    print('#{} {}'.format(tc, max_cnt))