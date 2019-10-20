import sys
sys.stdin = open('max_line.txt', 'r')

while True:
    try:
        t = int(input())
    except EOFError:
        break
    table = []
    for i in range(100):
        line = list(map(int, input().split()))
        table += [line]

    # row
    _sum = 0
    for i in range(100):
        for j in range(100):
            _sum += table[i][j]
        if max < _sum: max = _sum
        _sum = 0
    
    # column
    _sum = 0
    for i in range(100):
        for j in range(100):
            _sum += table[j][i]
        if max < _sum: max = _sum
        _sum = 0
    
    # diagonal
    for i in range(100):
        _sum += table[i][99-i]
    if max < _sum: max = _sum
    _sum = 0