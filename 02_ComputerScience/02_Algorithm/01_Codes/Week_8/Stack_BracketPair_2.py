import sys
sys.stdin = open('Stack_BracketPair_2.txt', 'r')


def get_idx(_list, element):
    for i in range(len(_list)):
        if _list[i] == element:
            return i


TC = int(input())
for tc in range(1, TC+1):
    _input = input()
    stack = [0] * len(_input)

    top = - 1
    push_s = '{('
    pop_s = '})'

    pushed = ''
    for s in _input:
        if s in push_s:
            top += 1
            stack[top] = s
            pushed = s
        elif s in pop_s and get_idx(pop_s, s) == get_idx(push_s, pushed):
            top -= 1

    if top == -1:
        print('#{} {}'.format(tc, 1))
    else:
        print('#{} {}'.format(tc, 0))