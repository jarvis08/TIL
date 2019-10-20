# Encryption & Compression

## 암호화, Encryption

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

<br>

<br>

## 압축, Compression

- Run-length encoding 알고리즘

  같은 값이 몇 번 반복되는가를 나타내어 압축

  RRRRRGGGBRRR -> R5G3B1R3

  - BMP 파일 포맷인 이미지 파일 포맷에 사용

- 허프만 코딩 알고리즘