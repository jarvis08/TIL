# counting sort 이용
T = int(input())
for tc in range(1, T+1):
    N = int(input())
    cards = input()
    cnt = [0] * 10

    for i in range(N):
        cnt[int(cards[i])] += 1
    
    maxl = 0
    for i in range(10):
        if cnt[maxl] <= cnt[i]:
            maxl = i

    print('#%d %d %d' %(tc, maxl, cnt[maxl]))