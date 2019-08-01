import sys
sys.stdin = open('flatten_input.txt', 'r')


def find_idx(n_list):
    max_idx = 0
    min_idx = 0
    for i in range(1, len(n_list)):
        if n_list[i] > n_list[max_idx]:
            max_idx = i
        elif n_list[i] < n_list[min_idx]:
            min_idx = i
    return max_idx, min_idx


for T in range(1, 11):
    D = int(input())
    boxes = list(map(int, input().split()))
    for d in range(D):
        max_idx, min_idx = find_idx(boxes)
        if boxes[max_idx] == boxes[min_idx]:
            print('#{} 0'.format(T))
            break
        boxes[max_idx] -= 1
        boxes[min_idx] += 1
    else:
        max_idx, min_idx = find_idx(boxes)
        print('#{} {}'.format(T, boxes[max_idx] - boxes[min_idx]))