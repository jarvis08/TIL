import sys
sys.stdin = open('Postfix_Notation.txt', 'r')


for tc in range(1, 11):
    _ = int(input())
    ex = input()

    stack_o = []
    postfix = []
    numbers = '0123456789'
    for s in ex:
        if s == '(':
            stack_o.append(s)
        elif s == ')':
            while stack_o[-1] != '(':
                postfix.append(stack_o.pop())
            stack_o.pop()

        elif s == '*':
            # 현재 순서의 연산자가 `Stack.top` 보다 우선순위가 높을 시 Push
            if len(stack_o) and stack_o[-1] == '*':
                postfix.append(s)
            else:
                stack_o.append(s)

        # 현재 순서의 연산자가 `Stack.top` 보다 우선순위가 같거나 낮으면 `Stack.top`을 출력 후 `push()`
        elif s == '+':
            if not len(stack_o):
                stack_o.append(s)
            elif stack_o[-1] == '(':
                stack_o.append(s)
            elif stack_o[-1] == '+' or stack_o[-1] == '*':
                postfix.append(stack_o.pop())
                stack_o.append(s)
        elif s in numbers:
            postfix.append(s)
    while len(stack_o):
        postfix.append(stack_o.pop())
    
    
    stack_n = []
    numbers = '0123456789'
    for s in postfix:
        if s in numbers:
            stack_n.append(int(s))
        elif s == '+':
            n_1 = stack_n.pop()
            n_2 = stack_n.pop()
            stack_n.append(n_1 + n_2)
        elif s == '*':
            n_1 = stack_n.pop()
            n_2 = stack_n.pop()
            stack_n.append(n_1 * n_2)
    print('#{} {}'.format(tc, stack_n.pop()))
