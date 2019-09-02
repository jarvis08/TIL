import sys
sys.stdin = open('Shuffle-O-Matic.txt', 'r')


TC = int(input())
for tc in range(1, TC+1):
    N = int(input())
    cards = list(map(int, input().split()))
    if sorted(cards) == cards:
        print('#{} {}'.format(tc, 0))
        continue

    result = 0
    for k in range(1, 6):
        half = int(N/2)
        p_1 = cards[:half]
        p_2 = cards[half:]

        for i in range(half):
            for j in range(half):
                if p_1[i] < p_2[j]:


        if sorted(cards) == cards:
            result = k
            break


