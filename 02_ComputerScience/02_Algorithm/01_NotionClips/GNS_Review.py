# Counting Sort를 이용하여 정렬해야 advanced level
p = ["ZRO","ONE","TWO","THR","FOR","FIV","SIX","SVN","EGT","NIN"]


def getidx(num):
    for i in range(10):
        # 일부러 넣어둔 오탈자의 단어인지를 확인
        if num[0] == p[i][0] and num[1] == p[i][1] and num[2] == p[i][2]:
            return i


TC = int(input())
for tc in range(1, TC+1):
    temp = input()
    nums = input().split()

    cnt = [0] * 10
    for num in nums:
        cnt[getidx(num)] += 1
    
    ans = ''
    for i in range(10):
        ans += p[i] * cnt[i]
    print('#{}'.format(tc), ans)


# Proffetional Level에서는 위 Counting 정렬을 최적화
# 인덱스 찾는 과정을 최적화
# 문자를 정확히 구분하기 위해 두 문자 이상을 비교
# 100 X 100의 공간에 10가지 숫자에 해당하는 공간에만 값을 집어 넣어둠
# ord의 알파멧에 해당하는 값이 100을 넘지 않기 때문에 100으로 설정
# 인덱스로 찾는 방법이기 때문에, 텍스트를 비교하여 0~9까지 비교하는 것보다 훨씬 빠름
numidx = [[0] * 100 for _ in range(100)]
numidx[ord('Z')][ord('R')] = 0
numidx[ord('O')][ord('N')] = 1
numidx[ord('T')][ord('W')] = 2
numidx[ord('T')][ord('H')] = 3
numidx[ord('F')][ord('O')] = 4
numidx[ord('F')][ord('I')] = 5
numidx[ord('S')][ord('I')] = 6
numidx[ord('S')][ord('V')] = 7
numidx[ord('E')][ord('G')] = 8
numidx[ord('N')][ord('I')] = 9

p = ["ZRO","ONE","TWO","THR","FOR","FIV","SIX","SVN","EGT","NIN"]
TC = int(input())
for tc in range(1, TC+1):
    temp = input()
    nums = input().split()
    
    cnt = [0] * 10
    for num in nums:
        cnt[numidx[ord(num[0])][ord(num[1])]] += 1
    
    ans = ''
    for i in range(10):
        ans += p[i] * cnt[i]
    print('#{}'.format(tc), ans)
