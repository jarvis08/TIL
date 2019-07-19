import math
print(math.sqrt(2))

def my_sqrt(n):
    x, y = 1, n
    result = 1
    while abs(result**2 - n) > 0.0000001:
        result = (x+y) / 2
        if result ** 2 < n:
            x = result
        else:
            y = result
    return result

print(my_sqrt(2))
