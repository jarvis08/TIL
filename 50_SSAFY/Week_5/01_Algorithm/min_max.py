import sys


sys.stdin = open('min_max_input.txt', 'r')


def min_max(nums):
    max_num = nums[0]
    min_num = nums[0]
    for num in nums:
        if num > max_num:
            max_num = num
        elif num < min_num:
            min_num = num
    return max_num - min_num


T = int(input())
for T in range(1, T+1):
    N = int(input())
    nums = list(map(int, input().split()))
    print('#{} {}'.format(T, min_max(nums)))