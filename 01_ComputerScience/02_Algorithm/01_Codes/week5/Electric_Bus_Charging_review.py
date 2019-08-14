N = 10
charger = list(map(int, input().split()))
stations = [0]*10
for idx in charger:
    stations[idx] = 1
K = 3
cnt = cur = 0
while True:
    pre = cur
    cur += K
    if cur >= N:
        break
    if stations[cur] == 1:
        cnt += 1
    else:
        for i in range(1, K+1):
            if stations[cur - i ] == 1:
                cur -= i
                cnt += 1
                break
        if cur == pre:
            cnt = 0
            break