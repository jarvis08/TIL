def quickSort(a, begin, end):
    # 시작 위치가 끝 위치보다 작은 경우만 시행, 아니라면 작업의 필요가 없다.
    if begin < end:
        # pivot을 기준으로 좌/우로 분할하여 따로 계산
        p = partition(a, begin, end)
        quickSort(a, begin, p-1)
        quickSort(a, p+1, end)
        

# pivot 값을 기준으로,
# 좌측에서는 pivot 보다 큰 값을 탐색
# 우측에서는 pivot 보다 작은 값을 탐색
# 두 값의 위치를 교환
def partition(a, begin, end):
    pivot = (begin + end) // 2
    L = begin
    R = end
    while L < R:
        while(a[L] < a[pivot] and L < R):
            L += 1
        while(a[R] >= a[pivot] and L < R):
            R -= 1
        # 만약 L<R이 성립되는 조건아래 교환할 값들을 찾지 못한다면,
        # L/R이 위치한 값과 Pivot의 위치를 바꿈
        if L < R:
            if L == pivot:
                pivot = R
            a[L], a[R] = a[R], a[L]
    a[pivot], a[R] = a[R], a[pivot]
    return R

a = [4, 7, 9, 12, 30, 50]
begin = 0
end = 6
quickSort(a, begin, end)