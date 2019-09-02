import sys
sys.stdin = open('Forth.txt', 'r')

TC = int(input())
for tc in range(1, TC+1):
    ex = input().split()
    stack = []
    
    for s in ex:
        if s.isdigit():
            stack.append(int(s))
        elif s != '.' and len(stack) < 2:
            print('#{} error'.format(tc))
            break
        elif s == '+':
            n2 = stack.pop()
            n1 = stack.pop()
            stack.append(n1 + n2)
        elif s == '-':
            n2 = stack.pop()
            n1 = stack.pop()
            stack.append(n1 - n2)
        elif s == '*':
            n2 = stack.pop()
            n1 = stack.pop()
            stack.append(n1 * n2)
        elif s == '/':
            n2 = stack.pop()
            n1 = stack.pop()
            stack.append(int(n1 / n2))
        elif s == '.':
            print('#{} {}'.format(tc, stack.pop()))
