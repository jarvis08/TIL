t = int(input())
for i in range(t):
    days = int(input())
    costs = list(map(int, input().split(' ')))
    max_idx = 0
    margin = 0    
    
    while True:
        max_idx = costs.index(max(costs))
        for i in range(max_idx):
            margin += costs[max_idx] - costs[i]
        for d in range(max_idx+1):
            del costs[0]
        if len(costs) <= 1:
            break
        if (days - 1) - (max_idx + 1) <= 0:
            break
    print('#{} {}'.format(i+1, margin))