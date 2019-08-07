def sequentialSearch(a, n, key):
    i = 0
    while i < n and a[i]!=key:
        i += 1
        if i < n:
            return i
        # 못 찾았을 경우
        return -1