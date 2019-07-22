def plus(n1, n2):
    return n1 + n2

def minus(n1, n2):
    return n1 - n2

def multiply(n1, n2):
    return n1 * n2

def devide(n1, n2):
    try:
        return n1 / n2
    except ZeroDivisionError:
        return '0으로 나눌 수 없습니다.'