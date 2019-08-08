def binarySearch(a, key):
    start = 0
    end = len(a) - 1
    while start <= end:
        middle = (start + end) // 2
        print("start : ", start)
        print("end : ", end)
        print("middle : ", middle)
        if a[middle] == key:
            return True
        elif a[middle] > key:
            end = middle - 1
        else:
            start = middle + 1
        print('---------------')
    return False

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 21, 23]
print(binarySearch(a, 20))