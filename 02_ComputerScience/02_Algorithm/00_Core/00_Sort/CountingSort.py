# A: input array
# k: maximum value of A
def counting_sort(origin, k):
    
    # init with -1
    _sorted = [-1] * len(origin)
    
    # init with zeros
    count = [0] * (k + 1)
    
    # count occurences
    for a in origin:
        count[a] += 1
    
    # update C
    # 앞 인덱스의 개수만큼 뒤로 밀려남
    for i in range(k):
        count[i+1] += count[i]
    
    # 가장 큰 수부터 _sorted의 뒷부분에 채워 넣기
    # count 리스트의 개수 -1 
    # 실질적으로 count 리스트의 개수는 _sorted의 index 역할
    for j in reversed(range(len(origin))):
        print('j\t:', j)
        _sorted[count[origin[j]] - 1] = origin[j]
        count[origin[j]] -= 1
        print('count\t:', count)   
        print('_sorted\t:', _sorted)   

    return _sorted

origin = [1, 2, 3, 1, 2, 3, 8, 10, 12]
print('result\t:', counting_sort(origin, 12))