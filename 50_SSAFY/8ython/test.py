import math
def like_sqrt(x):
    r = x
    l = float(x / 2)
    while True:
        if l ** 2 > x:
            l = l / 2
        else:
            break

    for i in range(50):
        half = (l+r)/2
        if half ** 2 > x:
            r = half
            continue
        elif half ** 2 < x:
            l = half
            continue
        break
    return l, r
print(like_sqrt(2))
print('오차 = ', math.sqrt(2) - like_sqrt(2)[0])