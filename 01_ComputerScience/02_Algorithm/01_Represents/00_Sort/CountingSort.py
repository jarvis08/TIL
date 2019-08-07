# A: input array
# k: maximum value of A
def counting_sort(A, k):
    
    # B: output array
    # init with -1
    B = [-1] * len(A)
    
    # C: counting array
    # init with zeros
    C = [0] * (k + 1)
    
    # count occurences
    for a in A:
        C[a] += 1
    
    # update C
    for i in range(k):
        C[i+1] += C[i]
    
    # update B
    for j in reversed(range(len(A))):
    	B[C[A[j]] - 1] = A[j]
    	C[A[j]] -= 1

    return B