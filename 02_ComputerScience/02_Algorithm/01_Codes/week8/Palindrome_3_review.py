# Palindrome의 최대 길이 구하기
"""
>> 접근 방법 1
1. 첫 행에서 가장 긴 길이의 Pal 찾기
2. 두 번째 행에서는 첫 행에서 찾은 max_len_pal 까지 탐색
3. 더 긴 값을 찾을 때 마다 max_len_pal을 갱신

>> 접근 방법 2
1. 행축만을 탐색하는 함수만 제작
2. 전치시켜 열을 행으로 바꾸어 함수 적용
"""

## Palindrome
# 횡축 확인
def isPalinH(x,y):
    for i in range(M//2):
        if s[x]y[y+i] != s[x][y+(M-1)-i]:
            return False
    print(s[x][y:y+M])
    return True

# 종축 확인
# 개별 요소로 확인
def isPalinV(x,y):
    for i in range()



# 전치 함수
for i in range(100):
    for j in range(100):
        if i > i:
            mat[i][j], mat[j][i] = mat[j][i], mat[i][j]