def binarySearch2(a, low, high, key):
    if low > high:
        return False
    else:
        middle = (low + high) // 2
        if key == a[middle]:
            return True
        elif key < a[middle]:
            return binarySearch2(a, low, middle - 1, key)
        elif a[middle] < key:
            return binarySearch2(a, middle + 1, high, key)

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 21, 23]
print(binarySearch2(a, 0, len(a) - 1, 20))