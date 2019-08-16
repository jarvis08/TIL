# Pattern Matching

> 고지식한 패턴 검색 알고리즘, Brute Force
>
> 카프-라빈 알고리즘
>
> KMP 알고리즘
>
> 보이어-무어 알고리즘

### 고지식한 패턴 검색 알고리즘, Brute Force

```python
def BruteForce(t, p):
    len_t = len(t)
    len_p = len(p)
    # t의 인덱스
    i = 0
    # p의 인덱스
    j = 0
    
    # 원본이 끝나거나 일치 지점을 찾을 때 까지
    while i < len_t and j < len_p:
        ## 패턴이 아닐 경우 위치를 돌려놓기
        # 만약 탐색 중 다르다고 판명된다면
        if t[i] != p[j]:
            # 탐색한 만큼 i 돌려놓기
            i = i - j
            # p를 처음(0)으로 돌려야 하는데, 이후에 +1 해줄 예정이므로 -1
            j = -1
        i += 1
        j += 1
    # j가 p 길이가 될 때까지 값이 같았다면
    if j == len_p:
        # 탐색하느라 증가한 i를 탐색한 만큼 낮추어 return
        return i - len_p
    else:
        # 통틀어서 발견하지 못했다면
        return -1
    
    
# 전체 텍스트
t = 'This is a book!'
# 찾을 패턴
p = 'is'
print(BruteForce(t, p))
```

- 최악의 경우 텍스트의 모든 위치에서 패턴을 비교

  시간 복잡도 = O(MN)

### KMP Algorithm

- 불일치가 발생한 텍스트 스트링의 앞 부분에 어떤 문자가 있는가를 미리 알고 있으므로,

  불일치가 발생한 앞 부분에 대하여 다시 비교하지 않고 매칭을 수행

- 패턴을 전처리하여 배열 `next[M]`을 구하여 잘못된 시작을 최소화

  - `next[M]` : 불일치가 발생했을 경우 이동할 다음 위치

- 시간 복잡도 = O(M + N)

- 방법

  1. Preprocessing

     (접두어 기준/접미어 기준)을 나열 및 비교하여, 겹치는 부분의 길이가 최대인 것을 탐색

  2. (패턴길이 - 최대 길이)만큼을 shifting 하며 탐색

### Boyer-Moore Algorithm

- 패턴의 뒤에서부터 비교 시작

- 대부분의 상용 SW에서 채택하는 알고리즘

- 패턴 문자가 일치하지 않을 시, 패턴의 길이만큼을 shift

- 오른쪽 끝 문자가 불일치하고, 비교했던 본문의 문자가 패턴내에 존재할 경우

  패턴에서 일치하는 문자를 찾아서 둘의 위치를 맞춰서 비교

- 최악의 경우 example

  본문 = 'aaaaa...baa'

  패턴 = 'baa'

  a가 계속해서 있으므로  패턴을 모든 a에 맞춰서 비교

### 문자열 매칭 알고리즘 비교

- 찾고자 하는 문자열 패턴의 길이를 m, 총 문자열의 길이를 n이라 할 때

  - 고지식한 패턴 검색 알고리즘

    수행시간 = O(mn)

  - 카프-라빈 알고리즘

    수행시간 = O(n)

---

## Index 빠르게 찾기

- e.g., Counting Sort로 적용하여 문제를 풀 때,

  정수가 아닌 list의 index를 부여

  ```python
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
  ```

  

