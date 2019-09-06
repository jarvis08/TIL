import sys
sys.stdin = open('DoublyLinkedList_MergeLists.txt', 'r')


TC = int(input())
for tc in range(1, TC+1):
    class Node():
        def __init__(self, data):
            self.data = data
            self.pre = None
            self.nxt = None


    class DoublyLinkedList():
        def __init__(self):
            dummy = Node('dummy')
            self.head = dummy
            self.tail = dummy
            self.n_datas = 0


        def dll_append(self, data):
            new = Node(data)
            new.pre = self.tail
            self.tail.nxt = new
            self.tail = new
            self.n_datas += 1
                

        def insertList(self, datas):
            nt = self.head
            for _ in range(self.n_datas):
                nt = nt.nxt
                if nt.data > datas[0]:
                    for i in range(len(datas)):
                        new = Node(datas[i])
                        nt.pre.nxt = new
                        new.pre = nt.pre
                        new.nxt = nt
                        nt.pre = new
                        self.n_datas += 1
                    break
            else:
                for data in datas:
                    self.dll_append(data)
            

        def __str__(self):
            result = []
            temp = self.tail
            for i in range(10):
                result.append(str(temp.data))
                temp = temp.pre
                if not temp:
                    break
            return ' '.join(result)


    N, M = map(int, input().split())
    datas = list(map(int, input().split()))
    
    dll = DoublyLinkedList()
    for i in range(N):
        dll.dll_append(datas[i])

    for _ in range(M-1):
        dll.insertList(list(map(int, input().split())))
    
    print('#{} {}'.format(tc, dll))
    