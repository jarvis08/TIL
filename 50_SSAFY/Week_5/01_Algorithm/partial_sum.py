import sys
sys.stdin = open('partial_sum_input.txt', 'r')

T = int(input())
for t in range(1, T+1):
    temp = list(map(int, input().split()))
    N, M = temp[0], temp[1]
    nums = list(map(int, input().split()))

    min_sum = 0
    max_sum = 0
    for i in range(N-M+1):
        to_sum = nums[i:i+M]
        summed = 0
        for n in to_sum:
            summed += n

        if i == 0:
            min_sum = summed
            max_sum = summed
        elif summed < min_sum:
            min_sum = summed
        elif summed > max_sum:
            max_sum = summed
    print('#{} {}'.format(t, max_sum - min_sum))
