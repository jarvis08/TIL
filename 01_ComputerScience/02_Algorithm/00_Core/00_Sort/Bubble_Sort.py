a = [7, 55, 12 , 44, 49]
def BubbleSort(a):
    # 4부터 0까지, 비교할 마지막 원소의 index를 의미
    for i in range(len(a)-1, 0, -1):
        # 0부터 i까지, 인접 원소인 [j], [j+1]을 마지막 원소인 [i]까지 비교
        for j in range(0, i):
            if a[j] > a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
    return a
print(BubbleSort(a))
# i 당 5, 4, 3, 2, 1회씩 j를 반복
# n(n+1)/2 = (n^2 + n)/2 => O(n^2)