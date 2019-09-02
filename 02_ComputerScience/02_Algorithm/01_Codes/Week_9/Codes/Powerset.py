# {1~10} 정수 원소의 합이 10인 부분 집합을 구하라
def backtrack(k, _sum):
    global cnt
    cnt += 1
    if k == N:
        if _sum == 10:
            for i in range(1, 11):
                if a[i] == True:
                    print(i, end=' ')
            print()
    else:
        k += 1
        a[k] = 1
        backtrack(k, _sum+k)
        a[k] = 0
        backtrack(k, _sum)


def prunning_backtrack(k, _sum):
    global cnt
    cnt += 1
    if k == N:
        if _sum == 10:
            for i in range(1, 11):
                if a[i] == True:
                    print(i, end=' ')
            print()
    else:
        k += 1
        if _sum + k <= 10:
            a[k] = 1
            prunning_backtrack(k, _sum+k)
        a[k] = 0
        prunning_backtrack(k, _sum)


N = 10
a = [0] * (N+1)

cnt = 0
backtrack(0, 0)
print("backtrack cnt :", cnt)
cnt = 0

prunning_backtrack(0, 0)
print("prunning_backtrack cnt :", cnt)


"""result
1 2 3 4
1 2 7
1 3 6
1 4 5
2 3 5
2 8
3 7 
4 6 
10
backtrack cnt : 2047
1 2 3 4
1 2 7
1 3 6
1 4 5
2 3 5
2 8
3 7 
4 6 
10
prunning_backtrack cnt : 250"""
