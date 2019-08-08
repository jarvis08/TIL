import sys
sys.stdin = open('sort_strangely.txt', 'r')


def sort_from_max(l):
    _max = l
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            if _max[i] < _max[j]:
                _max[i], _max[j] = _max[j], _max[i]
    return _max


T = int(input())
for t in range(1, T+1):
    n = int(input())
    nums = list(map(int, input().split()))
    ascent_sorted = sort_from_max(nums)
    for i in range(5):
        if i == 0:
            print('#{} {} {}'.format(t, ascent_sorted[i], ascent_sorted[n-1-i]), end='')
        elif i == (n//2)-1:
            print(' {} {}'.format(ascent_sorted[i], ascent_sorted[n-1-i]))
        else:
            print(' {} {}'.format(ascent_sorted[i], ascent_sorted[n-1-i]), end='')