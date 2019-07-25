func = lambda x : x ** 2
print(func(2))


a = [1, 2]
b = [3, 4]

print(list(map(lambda x, y : x + y, a, b)))