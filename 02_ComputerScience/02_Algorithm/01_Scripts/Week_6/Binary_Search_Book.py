import sys
sys.stdin = open('BinarySearch_book.txt', 'r')


def binarySearch(start, end, target):
    cnt = 0
    while True:
        cnt += 1
        half = int((start+end)/2)
        if half == target:
            return cnt
        if half < target:
            start = half
        if half > target:
            end = half


T = int(input())
for t in range(1, T+1):
    book, a, b = tuple(map(int, input().split()))
    a_cnt = binarySearch(1, book, a)
    b_cnt = binarySearch(1, book, b)
    if a_cnt == b_cnt:
        print('#{} 0'.format(t))
    if a_cnt > b_cnt:
        print('#{} B'.format(t))
    if a_cnt < b_cnt:
        print('#{} A'.format(t))
