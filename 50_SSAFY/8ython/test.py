t = int(input())
for i in range(1, t+1):
    nums = list(map(int, input().split(' ')))
    result = round(sum(nums) / 10)
    print('#' + str(i) + ' ' + str(result))