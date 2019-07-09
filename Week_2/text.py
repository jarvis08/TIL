# text.py
with open('ssafy.txt', 'w', encoding='utf-8') as f:
    for i in range(5):
        f.write('hell ssafy 가즈앙\n')

with open('ssafy.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        print(line.replace('\n', ''))

with open('problem.txt', 'w') as f:
    for i in range(4):
        f.write(str(i) + '\n')

lines = []
with open('problem.txt', 'r') as f:
    lines = f.readlines()

with open('problem.txt', 'w') as f:
    for i in range(4):
        f.write(str(lines[-i]))

lines.reverse()
with open('reverse.txt', 'w') as f:
    f.writelines(lines)