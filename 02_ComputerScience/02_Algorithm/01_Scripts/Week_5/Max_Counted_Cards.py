import sys
sys.stdin = open('cards_input.txt', 'r')

T = int(input())
for t in range(1, T+1):
    N = int(input())
    cards = input()
    cards_list = []
    count = dict()
    for card in cards:
        if card not in count.keys():
            count[card] = 1
        else:
            count[card] += 1

    max_card, max_count = '', 0
    for k, v in count.items():
        if v > max_count:
            max_card = k
            max_count = v
        elif v == max_count:
            if int(k) > int(max_card):
                max_card = k
    print('#{} {} {}'.format(t, max_card, max_count))
