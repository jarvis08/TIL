import sys
sys.stdin = open('StringOverlap.txt', 'r')

TC = int(input())
for tc in range(1, TC+1):
    string = input()
    while True:
        new_s = []
        previous = ''
        cnt = 0

        for i in range(len(string)):
            if string[i] == previous:
                cnt += 1
                previous = ''
                continue
            elif i == len(string) - 1:
                new_s += previous
                new_s += string[i]
            else:
                new_s += previous
                previous = string[i]
        if not cnt:
            break
        string = new_s
    print('#{} {}'.format(tc, len(string)))