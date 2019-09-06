import sys
sys.stdin = open('LinkedList_AddNums.txt', 'r')


TC = int(input())
for tc in range(1, TC+1):
    class Node():
        def __init__(self, data):
            self.data = data
            self.link = None


    class LinkedList():
        def __init__(self):
            dummy = Node('dummy')
            self.head = dummy
            self.tail = dummy


        def ll_append(self, data, index=None):
            new = Node(data)
            if index == 0 or index:
                prev = self.head
                for _ in range(index):
                    prev = prev.link
                new.link = prev.link
                prev.link = new
            else:
                self.tail.link = new
                self.tail = new


        def read(self, index):
            cur = self.head.link
            for _ in range(index):
                cur = cur.link
            return cur.data


        def __str__(self):
            result = []
            temp = self.head.link
            while True:
                result.append(temp.data)
                if not temp.link:
                    break
                temp = temp.link
            return str(result)


    N, M, L = map(int, input().split())
    datas = list(map(int, input().split()))
    ll = LinkedList()
    for data in datas:
        ll.ll_append(data)

    add = []
    for i in range(M):
        add.append(list(map(int, input().split())))
    for data in add:
        ll.ll_append(data[1], data[0])
    
    print('#{} {}'.format(tc, ll.read(L)))
        
