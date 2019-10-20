# String

## 문자열의 표준

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

- 문자열의 분류

  - Fixed Length

  - Variable Length

    - Length Controlled

      e.g., Java

    - Delimited

      끝내는 Marker가 필요

      Null Character(`\0`, `00000000` bit열) 을 사용

      e.g., C

<br>

<br>

## 문자열 처리

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