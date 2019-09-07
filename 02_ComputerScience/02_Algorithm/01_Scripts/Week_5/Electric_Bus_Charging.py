import sys
sys.stdin = open('electric_bus_input.txt', 'r')


def findCharger(charger_list, present, endpoint):
    for c in charger_list:
        if c > present:
            return c
    return endpoint


T = int(input())
for T in range(1, T+1):
    K, N, M = tuple(map(int, input().split()))
    charger = list(map(int, input().split()))
    energy = K
    count = 0
    for i in range(1, N):
        energy -= 1
        if energy < 0:
            print('#{} 0'.format(T))
            break
        if i not in charger:
            continue

        next_charger = findCharger(charger, i, N)
        if next_charger - i > K:
            count = 0
            break
        elif next_charger - i <= energy:
            continue
        else:
            energy = K
            count += 1
    print('#{} {}'.format(T, count))
