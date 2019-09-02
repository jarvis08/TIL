def BruteForce(t, p):
    i = 0
    j = 0
    N = len(t)
    M = len(p)
    while j < M and i < N:
        if t[i] != p[j]:
            i = i - j
            j -= 1
        i += 1
        j += 1
    if j == M: return 1
    else: return 0