import sys
sys.stdin = open('DP_Rectangle_Area.txt', 'r')


A = 10
B = 20
C = 20


def factorial(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result


TC = int(input())
for tc in range(1, TC+1):
    N = int(input())
    cnt = 0
    for a in range((N//A) + 1):
        for b in range((N//B) + 1):
            for c in range((N//C) + 1):
                if A*a + B*b + C*c > N:
                    break
                if A*a + B*b + C*c == N:
                    cnt += int(factorial(a+b+c) / (factorial(a)*factorial(b)*factorial(c)))
                    
    print('#{} {}'.format(tc, cnt))