# SSAFY Week7 Day3

---

## Intermediate Reivew

1. 색칠하기

   필요한 영역을 모두 `+1`하여 최대값의 요소들을 Count

2. 부분집합의 합

   **Binary Counting** 응용

3. 금속 막대

   ```python
   for i in range(1, N-1):
       for j in range(i, N):
           if arr[i * 2 - 1] == arr[j * 2]:
               arr[i*2], arr[j*2] = arr[j*2], arr[i*2]
               arr[i*2], arr[j*2+1] = arr[j*2+1], arr[i*2]
               break
   ```

   

---

## Under-bar( _ )를 사용하는 경우

- 무의미한 변수를 할당해야 할 때

  - for문의 i가 사용되지 않을 경우 _로 설정

  - tuple의 요소를 개별로 저장하는데, 사용되지 않는 요소의 경우 _로 설정

- Built-in 함수의 이름을 사용하고 싶을 때

  - `_sum = 0``
  - ``_int = result`

---

## 행렬곱은 순서에 따라 계산량이 다르다.

- `[2][3] X [3][4] X [4][5]` 세 행렬을 곱할 때,

  1. 앞에서 부터 순서대로

     - `2*3*4 = 24`
     - `2*4*5 = 40`

     총 64 회 계산

  2. 뒤의 행렬 부터

     - `3*4*5 = 60`

     - `2*3*5 = 30`

     총 90 회 계산

---

## String

- ASCII, American Standard Code for Information Interchange

  - 표준 ASCII

    7 bit 인코딩으로 128개의 문자를 표현

    - 128 = (33개의 출력 불가능한 제어 문자) + (공백을 비롯한 95개의 출력 가능한 문자)

  - 확장 ASCII

    표준 문자 이외의 악센트 문자, 특수 문자, 특수 기호 등 부가적인 문자 128개를 추가 지원

    - 1B 내의 8bit를 모두 사용

- 언어의 정류

  - 조합형 언어

    한글, `ㄱ` + `ㅏ` + `ㅇ` = `강`

  - 완성형 언어

    중국어

- Unicode

  국가간 표준을 맞추기 위해 제작

  - 유니코드 인코딩, UTF(Unicode Transformation Format)
    - UTF-8
      - in Web
      - Min : 8bit
      - Max : 32bit(1 Byte * 4)
    - UTF-16
      - in Windows, Java
      - Min : 16bit
      - Max : 32bit(2 Byte * 4)
    - UTF-32
      - in Unix
      - Min : 32bit
      - Max : 32bit(4 Byte * 1)

- Python Encoding

  - 2.x : ASCII
  - 3.x : UTF-8
  - Encoding 방식 변경 시 코드 첫 줄에 기입

- 문자열, String

  - Fixed Length

  - Variable Length

    - Length Controlled

      e.g., Java

    - Delimited

      끝내는 Marker가 필요

      Null Character(`\0`, `00000000` bit열) 을 사용

      e.g., C

- C 언어에서의 문자열 처리

  - ASCII
    - `strlen('홍길동') = 6`
  - char들의 배열로 저장하는 응용 자료형

  - Null Character인 `\0`을 마지막에 필수적으로 작성

  - 문자열 처리에 필요한 연산을 함수 형태로 제공

    `strlen()`, `strcpy()`, `strcmp()`, `atoi()`, `itoa()`

    - `atoi()` : 문자 타입을 정수 타입으로 변환
    - `itoa()` : 정수 타입을 문자 타입으로 변환

  ```c
  // 아래 line들은 동일한 string을 표현
  char ary[] = {'a', 'b', 'c', '\0'};
  char ary[] = "abc";
  ```

- Java(객체지향)에서의 문자열 처리

  - 유니코드(UTF16, 2byte)
    - `strlen('홍길동') = 3`
  - String Class 사용

- Python에서의 문자열 처리

  - 유니코드(UTF8)
    - `strlen('홍길동') = 3`
  - `char` type이 존재하지 않음
  - 모든 Text Data는 String으로 통일
  - `+` : 연결
  - `*` : 반복
  - Iterable 하며, Immutable(수정 불가)
  - indexing, slicing 연산 가능

- Type Casting Library
  - C
    - `atoi()`
    - `itoa()`
  - Java
    - `Integer.parseInt(String)`
    - `toString()`
  - Python
    - `int()`
    - `float()`
    - `str()`
    - `repr()`
    - `ord('a')`  : a에 해당하는 Unicode 숫자를 반환
      - `ord('a') = 97`
      - `ord( '1') = 49`

---

## 패턴 매칭

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

### KMP 알고리즘

- 불일치가 발생한 텍스트 스트링의 앞 부분에 어떤 문자가 있는가를 미리 알고 있으므로,

  불일치가 발생한 앞 부분에 대하여 다시 비교하지 않고 매칭을 수행

- 패턴을 전처리하여 배열 `next[M]`을 구하여 잘못된 시작을 최소화

  - `next[M]` : 불일치가 발생했을 경우 이동할 다음 위치

- 시간 복잡도 = O(M + N)

- 방법

  1. Preprocessing

     (접두어 기준/접미어 기준)을 나열 및 비교하여, 겹치는 부분의 길이가 최대인 것을 탐색

  2. (패턴길이 - 최대 길이)만큼을 shifting 하며 탐색

### 보이어-무어 알고리즘

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

### 암호화

- 시저 암호, Caesar Cipher

  - 줄리어스 시저가 사용

  - 평문에서 사용되고 있는 알파벳을 일정한 문자 수 만큼 '평행이동' 시켜 암호화

  - 평행 키값 = 1 일 때는 카이사르 암호화라고 부른다.

    평문 = 'ABCD'

    암호문 = 'BCDE'

- 문자 변환표를 이용한 암호화(단일 치환 암호)

  랜덤하게 암호화, 가능한 키의 수는 26!

  'A' -> 'H'

  'B' -> 'C'

- bit열의 암호화

  배타적 논리합(exclusive-or, XOR) 연산 사용

### 문자열 압축

- Run-length encoding 알고리즘

  같은 값이 몇 번 반복되는가를 나타내어 압축

  RRRRRGGGBRRR -> R5G3B1R3

  - BMP 파일 포맷인 이미지 파일 포맷에 사용

- 허프만 코딩 알고리즘