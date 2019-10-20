import sys
sys.stdin = open('Matrix_Min_Sum.txt', 'r')

def back_track(i, j):
    for i in range(N):
        for j in range(N):
            
    visited[i][j]
    if found:
        return

TC = int(input())
for tc in range(1, TC+1):
    N = int(input())

    matrix = []
    for i in range(N):
        matrix += [list(map(int, input().split()))]

    visited = [[0] * N for _ in range(N)]