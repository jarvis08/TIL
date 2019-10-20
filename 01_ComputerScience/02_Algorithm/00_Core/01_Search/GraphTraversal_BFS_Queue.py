# 강의 자료 연습문제
def BFS(w):
    front = rear = -1
    r += 1
    q[r] = w

    print(q[rear])
    visited[w] = 1

    while front != rear:
        front += 1
        w = q[front]
        for i in range(3):
            if G[w][i] and not visited[G[w][i]]:
                rear += 1
                q[rear] = G[w][i]
                print(q[rear])
                visited[G[w][i]] = 1

G = [
    [0, 0 0],
    [2, 3, 0]
]
q = [0] * 10
visited = [0] * 10
BFS(1)