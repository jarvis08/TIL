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
    row = []
    column = []
    diagonal = [0, 0]
    for i in range(100):
        row += [0]
        column += [0]

    for i in range(100):
        for j in range(100):
            if i == j:
                diagonal[0] += table[i][j]
            elif i + j == 99:
                diagonal[1] += table[i][j]
            row[i] += table[i][j]
            column[i] += table[j][i]

    # max line 찾기
    overall = row + column + diagonal
    max_line = 0
    for s in overall:
        if max_line < s:
            max_line = s
    print('#{} {}'.format(t, max_line))