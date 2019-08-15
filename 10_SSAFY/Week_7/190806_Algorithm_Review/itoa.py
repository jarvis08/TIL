def itoa(x):
    str_reversed = ''
    while True:
        r = x % 10
        str_reversed = str_reversed + chr(r + ord('0'))
        x //= 10
        if x == 0:
            break
    
    string = ''
    for i in range(len(sr) - 1, -1, -1):
        string = string + str_reversed[i]
    
    return s

print(itoa(1234))