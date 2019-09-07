import sys
sys.stdin = open('Card_Deck.txt', 'r')


TC = int(input())
for tc in range(1, TC+1):
    yj = input()
    c = 0
    deck = {
        'S': [],
        'D': [],
        'H': [],
        'C': []
    }
    check_overlap = []
    overlapped = False
    for i in range(0, len(yj)-2, 3):
        if yj[i:i+3] in check_overlap:
            print('#{} ERROR'.format(tc))
            overlapped = True
            break
        check_overlap += [yj[i:i+3]]
        
        shape = yj[i:i+1]
        num = yj[i+1:i+3]
        deck[shape] += [int(num)]
    if overlapped:
        continue
    
    cnt = {
        'S': 0,
        'D': 0,
        'H': 0,
        'C': 0
    }
    for shape in deck.keys():
        for i in range(1, 14):
            if i not in deck[shape]:
                cnt[shape] += 1
    print('#{} '.format(tc), end='')
    for v in cnt.values():
        print(v, end=' ')
    print()