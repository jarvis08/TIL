import sys
sys.stdin = open('bolt_connection.txt', 'r')

def find_unique(head, tail):
    n = len(head)
    uni_head_idx = 0
    uni_tail_idx = 0
    min_head = 5000
    min_tail = 5000
    for i in range(n):
        head_cnt = 0
        tail_cnt = 0
        for j in range(n):
            if i == j:
                continue
            if head[i] == tail[j]:
                head_cnt += 1
                continue
            if tail[i] == head[j]:
                tail_cnt += 1
        if min_head > head_cnt:
            min_head = head_cnt
            uni_head_idx = i
        if min_tail > tail_cnt:
            min_tail = tail_cnt
            uni_tail_idx = i
    return uni_head_idx, uni_tail_idx


def find_equal(tail, bolts):
    for i in range(len(bolts)):
        if tail == bolts[i][0]:
            return i
    else:
        return None


T = int(input())
for tc in range(1, T+1):
    n = int(input())
    temp = list(map(int, input().split()))
    bolts = []
    for i in range(0, len(temp) - 1, 2):
        bolts += [[temp[i], temp[i+1]]]

    # Find 1st and last index
    # find non-unique head
    head = []
    tail = []
    for i in range(len(bolts)):
        head += [bolts[i][0]]
        tail += [bolts[i][1]]

    h_idx, t_idx = find_unique(head, tail)
    h = bolts[h_idx]
    t = bolts[t_idx]
    if h_idx - t_idx > 0:
        del bolts[h_idx]
        del bolts[t_idx]
    else:
        del bolts[t_idx]
        del bolts[h_idx]

    connected = [h] + [0] * (n-2) + [t]
    tail_idx = 0
    head_idx = 1
    while len(bolts) > 0:
        head_idx = find_equal(connected[tail_idx][1], bolts)
        if head_idx == None:
            break
        connected[tail_idx+1] = bolts[head_idx]
        del bolts[head_idx]
        tail_idx += 1
    print('#{}'.format(tc), end=' ')
    for i in range(n):
        if i == n-1:
            print('{} {}'.format(connected[i][0], connected[i][1]))
        else:
            print('{} {} '.format(connected[i][0], connected[i][1]), end='')