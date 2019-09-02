import sys
sys.stdin = open('Queue_Extemely-Basic.txt', 'r')


class Queue():
    queue = []
    front = -1
    rear = -1
    size = 0


    def __init__(self, N):
        self.queue = [0]*N
        self.size = N


    def enqueue(self, value):
        if self.rear == self.size-1:
            self.rear = 0
        else:
            self.rear += 1
        self.queue[self.rear] = value
    

    def dequeue(self):
        temp = self.queue[self.front+1]
        if self.front == self.size-2:
            self.front = -1
        else:
            self.front += 1
        return temp


    def __str__(self):
        return '{}'.format(self.queue[self.front+1])


TC = int(input())
for tc in range(1, TC+1):
    N, M = map(int, input().split())
    N_list = list(map(int, input().split()))

    q = Queue(N)
    for n in N_list:
        q.enqueue(n)

    for i in range(M):
        nxt = q.dequeue()
        q.enqueue(nxt)
    print('#{} {}'.format(tc, q))