class Queue():
    q = []

    def enqueue(self, student):
        self.q.append(student)
    
    def dequeue(self):
        return self.q.pop(0)

        
q = Queue()
mizzu = 20
students = [0]*20
exist = True
k = 1
while exist:
    for i in range(k):
        print('>> {}번 학생이 줄을 선다.'.format(i+1))
        q.enqueue(students[i])
        if i == k-1:
            for j in range(k):
                n_zzu = q.dequeue()
                students[j] = n_zzu + 1
                mizzu -= 1
                if not mizzu:
                    print(j)
                    exist = False
                    break
    k += 1
    