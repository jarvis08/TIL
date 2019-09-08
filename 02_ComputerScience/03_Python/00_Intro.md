# Intro

**참고자료 file_path** : ./50_SSAFY/8ython/notes/01.python_intro

---

## Turing Completeness 

- 저장 :: 어디에 무엇을 어떻게 넣는가?
  1. 숫자
  2. 글자
  3. 참/거짓
- 조건
- 반복

------

## 기본 자료형

- 숫자
- 글자
- 참/거짓(boolean)

---

## Python은 어떻게 저장하는가?

- 변수 (variable) 

  :: 박스 1개

- 리스트 (list)

  :: 박스 여러개

  ​	list = ["강남구", ...]

- 딕셔너리 (dictionary)

  :: 견출지 붙인 박스들의 묶음

  ​	dic = {"강남구" : 50, ...}

---

- Python Tutor

  ​	: Visualize Code and Get Live Help

  ​	line by line으로 python 내부에서 어떻게 실행되는지 확인 가능

  ​	http://pythontutor.com/

  ​	http://pythontutor.com/visualize.html#mode=edit
  
- `$ python -i` : git bash에서 python interactive 실행

- `.ipynb ` : jupyter notebook 확장자, json 파일 형식으로 저장

- `PEP-8` : python enhancement proposal, python 개발자들의 표준안

- Linter : Code Style Guide Extention

---

- python keyword(식별자, 예약어)

  : `keyword` library에 포함되어 있다. 이 내용들은 함수/클래스/변수명 불가

  https://wikidocs.net/20557
  
  ```python
  import keyword
  keyword.kwlist
  ```

- docstring : `func.__doc__` 기입 시 docstring 내용의 string 출력

  ```python
  def mysum(a, b):
      """덧셈 함수다.
      이 줄은 실행이 되지 않는다.
      docstring을 쓰는 이유는 __doc__을 사용하기 때문이다.
      """
      return a + b
  print(mysum(2, 5))
  print(type(mysum.__doc__))
  print(mysum.__doc__)
  """결과
  7
  <class 'str'>
  덧셈 함수다.
      이 줄은 실행이 되지 않는다.
      docstring을 쓰는 이유는 __doc__을 사용하기 때문이다.
  """
  ```
  
- 기본적으로  `;` semi-colon을 사용하지 않는다.

  한 줄로 표기할 떄는 이를 사용 가능하다.

  ```python
  print("happy")
  print("hacking")
  '''결과
  happy
  hacking
  '''
  # 에러 유형
  print("happy")print("hacking")
  # 해결
  print("happy");print("hacking")
  ```

- sting 도중 `\` 없이 줄 나눔 하면 에러

  ```python
  # error(string)
  print('
        파이썬은 쉽다.
        파이썬은 강력하다.
        ')
  # ok(\ backslash 사용)
   print('\
        파이썬은 쉽다.\
        파이썬은 강력하다.\
        ')
  # list, dic은 그냥 ok
  matjip = {
      '짬뽕' : "베이징코야",
      '햄버거' : "바스버거",
      '닭도리탕' : '고갯마루',
      '부대찌개' : "대우식당",
      '돼지고기' : '백운봉 막국수'
  }
  ```

---

## 변수, Variable

- **Scope**
  - global(바깥)에서 제어문의 내부(local) 변수 참조 불가
  - 제어문 안에서는 바깥 변수를 참조 가능

- `id()` : python이 관리하는 공간의 위치 제공

  정말 동일한 것을 가리키는지 확인 가능

- 변수 할당

  ```python
  x = "ssafy"
  type(x)
  id(x)
  """결과
  str
  21655776
  """
  
  # 같은 값 동시에 할당
  y = z = 1004
  print(y, z)
  print(id(y), id(z))
  """결과
  1004 1004
  106876736 106876736
  """
  
  # 두개의 변수에 값 두개 할당
  name, age = 'dongbin', 28
  print(name, age)
  print(type(name), type(age))
  print(id(name), id(age))
  """결과
  dongbin 28
  <class 'str'> <class 'int'>
  """
  
  # 튜플로 지정하여 할당해도 같은 결과
  (names, ages) = ('dongbin', 28)
  print(name, age)
  print(type(name), type(age))
  """결과
  dongbin 28
  <class 'str'> <class 'int'>
  """
  ```

- 변수 값 swapping

  ```python
  # 변수 x와 y의 값 swapping
  x, y = 5, 10
  tmp = x
  x = y
  y = tmp
  print(x, y)
  """결과 10 5"""
  
  ### python은 tuple을 사용하므로 그냥 가능
  h, z = 5, 10
  h, z = z, h
  print(h, z)
  """결과 10 5"""
  ```

- arbitrary-precision arithmetic
  - 파이썬에서 아주 큰 정수를 표현할 때 사용하는 메모리의 크기 변화
  - 사용할 수 있는 메모리양이 정해져 있는 기존의 방식과 달리, 현재 남아있는 만큼의 가용 메모리를 모두 수 표현에 끌어다 쓸 수 있는 형태
  - 특정 값을 나타내는데 4바이트가 부족하다면 5바이트, 더 부족하면 6바이트까지 사용 할 수 있게 유동적으로 운용

  ```python
  # 파이썬에서 가장 큰 숫자를 활용하기 위해 sys 모듈 사용
  # 파이썬은 기존 C 계열 프로그래밍 언어와 다르게 정수 자료형에서 오버플로우가 없다.
  # arbitrary-precision arithmetic를 사용하기 때문
  import sys
  max_int = sys.maxsize
  print(max_int)
  
  big_num = max_int * max_int
  print(big_num)
  """결과
  2147483647
  4611686014132420609
  21267647892944572736998860269687930881
  """
  ```

- `string` 자료형으로 표현된 문자열이 숫자인지 판별하기

  ```python
  num = input()
  if num.isdigit():
      print('숫자입니다.')
  else:
      print('문자입니다.')
  ```

- n진수, `0 + (n진법표기) + (원하는숫자)`

- 진법 표기

  `0o` : 8진수

  `0b` : 2진수

  `0x` : 16진수

  ```python
  # 2진수 0b
  binary_num = 0b10
  print(binary_num)
  # 8진수 0o10
  octal_num = 0o10
  print(octal_num)
  # 10진수
  decimal_num = 10
  print(decimal_num)
  # 16진수 0x
  hexadecimal_num = 0x10
  print(hexadecimal_num)
  
  '''결과
  2
  8
  10
  16
  '''
  ```

- float : 부동 소수점(고정 소수점x), 소수점이 유동적

  ```python
  # e를 사용
  b = 314e-2
  print(type(b))
  print(b)
  '''결과
  <class 'float'>
  3.14
  '''
  ```

- 반올림

  `round(계산식, 몇재짜리까지)`

  ```python
  round(3.5 - 3.15, 2)
  '''결과
  0.35'''
  
  # 두 개의 값이 같은가
  3.5 - 3.15 == 0.35
  """결과
  False
  """
  
  # 1. 이에 대한 기본적인 처리방법
  # abs(a-b) <= 1e-10
  # 0.0000000001 까지의 오차 범위는 같고 처리
  a = 3.5 - 3.15
  b = 0.35
  abs(a-b) <= 1e-10
  """결과
  True
  """
  
  # 2. sys 모듈을 통해 처리
  # 원하는 값까지 충분히 가까운 값을 얻었나에 대한 판별
  # sys.float_info.epsilon
  import sys
  abs(a - b) <= sys.float_info.epsilon
  """결과
  True
  """
  
  # 3. python 3.5부터 활용 가능한 math 모듈을 통해 처리
  # math.isclose(a,b)
  import math
  math.isclose(a, b)
  """결과
  True
  """
  ```

- `math.isclose(a, b)`

  ``math.``isclose`(*a*, *b*, ***, *rel_tol=1e-09*, *abs_tol=0.0*)`

  rel_tol 값을 조정하면 더 까다롭게 비교 가능

### Under-bar( _ )를 사용하는 경우

- 무의미한 변수를 할당해야 할 때

  - for문의 i가 사용되지 않을 경우 _로 설정

  - tuple의 요소를 개별로 저장하는데, 사용되지 않는 요소의 경우 _로 설정

- Built-in 함수의 이름을 사용하고 싶을 때

  - `_sum = 0``
  - ``_int = result`

---

## 자료형

- **complex**, 복소수

  허수부를 j로 표현

  ```python
  # 허수부를 다르게 표현하면 허수 처리 안하므로, 1j 4j 등으로 표기
  a = 3 - 4j
  type(a)
  """결과
  complex
  """
  
  # 복소수와 관련된 메소드
  print(a.imag)
  print(a.real)
  """결과
  3.0
  -4.0
  """
  ```

- **Boolean**

  `True`와 `False`로 이뤄진 `bool` 타입이 있으며, 비교/논리 연산을 수행 등에서 활용

  ```python
  # True와 False의 타입
  print(bool(0))
  print(bool(1))
  print(bool(2))
  print(bool(None))
  print(bool([]))
  print(bool({}))
  print(bool(''))
  print(bool('hello'))
  """결과
  False
  True
  True
  False
  False
  False
  False
  True
  """
  ```

- **None**

  파이썬에서는 값이 없음을 표현하기 위한 `None`타입이 존재

  ```python
  # method는 입력값(input)과 반환값(return)이 존재
  # print는 input을 표시하는 기능이 있지만,
  # return 값이 존재하지 않기에 None을 출력
  print(print)
  print(print('hello'))
  """결과
  <built-in function print>
  hello
  None
  """
  
  # 원본을 변형시키는 sort()
  # 원본을 변형시킨 후 반환하는 기능이 없어 sort_num에는 None 값을 할당
  # 원본 data를 변형시키는 method를 destructive method 라고 부름
  sort_num = [5, 4, 3, 2, 1].sort()
  # 원본을 변형시키지 않으며 변형 값을 저장하는 sorted()
  sorted_num = sorted([5, 4, 3, 2, 1])
  print(sort_num)
  print(sorted_num)
  """result
  None
  [1, 2, 3, 4, 5]
  """
  ```

- **String**

  - 문자열은 Single quotes(`'`)나 Double quotes(`"`)을 활용하여 표현 가능하다.
  
  - 단, 문자열을 묶을 때 동일한 문장부호를 활용해야하며, `PEP-8`에서는 **하나의 문장부호를 선택**하여 유지하도록 권장(Pick a rule and Stick to it)
  
    ```python
      she_said = '김지수님은 말씀하셨다.'
      go_home = "오늘은 집에 빨리 가세요."
      print(she_said, go_home)
      print(she_said + ' ' + go_home)
      """result
      김지수님은 말씀하셨다. 오늘은 집에 빨리 가세요.
      김지수님은 말씀하셨다. 오늘은 집에 빨리 가세요.
      """
    ```
  
    ```python
      # 사용자에게 받은 입력은 기본적으로 str
      age = input("당신의 나이를 입력해 주세요.")
      print(age)
      print(type(age))
      """result
      당신의 나이를 입력해 주세요.28
      28
      <class 'str'>
      """
    ```
  
  - 여러줄에 걸쳐있는 문장은 excape code 혹은 `"""` 사용 가능
  
    `PEP-8`에 따르면 이 경우  `"""`를 사용할 것을 권장
  
    ```python
      # excape code를 이용하여 single quote 중복 사용하기
      she_said = '김지수님은 말씀하셨다.\'오늘은 집에 빨리 가세요.\''
      print(she_said)
      """result
      김지수님은 말씀하셨다.'오늘은 집에 빨리 가세요.'
      """
      # """ 사용하기
      print("""여러줄에 걸쳐있는 문장은 다음과 같이 표현 가능하다.
      PEP-8에 따르면, 이 경우에는 반드시 이렇게 하도록 한다.""")
    ```
  
    ```python
      # string concatenation 합체
      concat = "안녕하세요," + ' 저는' + ' 조동빈입니다.'
      # string interpolation(보간법) 수술(삽입)
      # python 3.6 이상 버전의 fstring
      name = '조동빈'
      inter_1 = f'안녕하세요, 저는 {name}입니다.'
      # 일반적인 python
      inter_2 = '안녕하세요, 저는 {}입니다.'.format(name)
      print(concat)
      print(inter_1)
      print(inter_2)
      
      """result
      안녕하세요, 저는 조동빈입니다.
      안녕하세요, 저는 조동빈입니다.
      안녕하세요, 저는 조동빈입니다.
      """
    ```

- `\` excape code 문자열

  | <center>예약문자</center> |   내용(의미)    |
  | :-----------------------: | :-------------: |
  |            \n             |     줄바꿈      |
  |            \t             |       탭        |
  |            \r             |   캐리지리턴    |
  |            \0             |    널(Null)     |
  |           `\\`            |       `\`       |
  |            \'             | 단일인용부호(') |
  |            \"             | 이중인용부호(") |

  ```python
    # print를 하는 과정의 이스케이프 문자열 활용
    print('가나다', end='')
    print('라마바', end='\0')
    print('사아자', end='\t')
    print('차카타')
    """result
    가나다라마바사아자	차카타"""
    
    # 물론, end 옵션은 이스케이프 문자열이 아닌 다른 것도 가능합니다.
    print('가나다', end='!')
    print('라마바', end='!')
    """result
    가나다!라마바!"""
  ```

- String Interpolation
  - `%-formatting`

  - [`str.format()`](https://pyformat.info/)

  - [`f-strings`](https://www.python.org/dev/peps/pep-0498/) : 파이썬 3.6 버전 이후에 지원 되는 사항

    ```python
    # %-formatting
    # s : string
    # i : int
    print('hello, I\'m %s' %name)
    print('hello, I\'m %s, and I live in %s' %(name, address))
    """result
    hello, I'm Dongbin
    hello, I'm Dongbin, and I live in Jamsil"""
    
    # str.format()
    print('hello, I\'m {}, and I live in {}'.format(name, address))
    # f-string
    print(f'hello, I\'m {name}, and I live in {address}')
    """result
    hello, I'm Dongbin, and I live in Jamsil"""
    ```

    ```python
    # interpolation 예제 with datetime
    import datetime
    today = datetime.datetime.now()
    print(f'오늘은 {today:%y}년 {today:%m}월 {today:%d}일 {today:%a}day {today.hour}시 {today.minute}분')
    
    year = datetime.datetime.now().year
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    print(f'오늘은 {year}년 {month}월 {day}일')
    
    """result
    오늘은 19년 07월 15일 Monday 14시 20분
    오늘은 2019년 7월 15일"""
    
    # 수식넣기
    pi = 3.141592
    radius = 2
    print(f'원주율이 {pi}일 때, 반지름이 {radius}인 원의 넓이는 {radius * radius * pi}다.')
    area = pi * radius
    print(f'원주율이 {pi}일 때, 반지름이 {radius}인 원의 넓이는 {area}다.')
    """result
    원주율이 3.141592일 때, 반지름이 2인 원의 넓이는 12.566368다.
    원주율이 3.141592일 때, 반지름이 2인 원의 넓이는 6.283184다."""
    ```

---

## 연산자

- **연산자**

  | 연산자 | 내용           |
  | ------ | -------------- |
  | +      | 덧셈           |
  | -      | 뺄셈           |
  | *      | 곱셈           |
  | /      | 나눗셈         |
  | //     | 몫             |
  | %      | 나머지(modulo) |
  | **     | 거듭제곱       |

  ```python
  print(5 / 2)
  print(int(5) / 2)
  print(int(5 / 2))
  print(5 // 2)
  print(5 % 2)
  print(2**10)
  """result
  2.5
  2.5
  2
  2
  1
  1024"""
  
  # divmod로 몫, 나머지 한번에 구하기
  quotient, remainder = divmod(5,2)
  print(quotient, remainder)
  """result
  2 1"""
  
  # 음수, 양수 전환
  pos = 4
  print(-pos)
  neg = -4
  print(-neg)
  """result
  -4
  4"""
  ```

- **비교 연산자**

  | 연산자 | 내용     |
  | ------ | -------- |
  | a > b  | 초과     |
  | a < b  | 미만     |
  | a >= b | 이상     |
  | a <= b | 이하     |
  | a == b | 같음     |
  | a != b | 같지않음 |

  ```python
  3 > 6
  3 == 4
  3 != 4
  3 == 3.0
  'hi' == 'Hi'
  """result
  False
  False
  True
  True
  False"""
  ```

- **논리 연산자**

  | 연산자  | 내용                         |
  | ------- | ---------------------------- |
  | a and b | a와 b 모두 True시만 True     |
  | a or b  | a 와 b 모두 False시만 False  |
  | not a   | True -> False, False -> True |

  우리가 보통 알고 있는 `&` `|`은 파이썬에서 비트 연산자이다.

  ```python
  # and
  print(True and True)
  print(True and False)
  print(False and True)
  print(False and False)
  """result
  True
  False
  False
  False"""
  
  # or 연산자
  print(True or True)
  print(True or False)
  print(False or True)
  print(False or False)
  """result
  True
  True
  True
  False"""
  
  # not
  print(not True)
  print(not 0)
  """result
  False
  True"""
  ```

- **단축평가**(short-circuit evaluation)

  and : a가 거짓이면 a를 리턴하고, 참이면 b를 리턴한다.

  or : a가 참이면 a를 리턴하고, 거짓이면 b를 리턴한다.

  - 단축 평가 활용 예시 : 빈 배열과 값이 있는 배열이 있을 때, 빈 배열은 무시되고 값이 있는 배열을 출력

  ```python
  # and 단축평가(short-circuit evaluation)
  print(3 and 5)
  print(3 and 0)
  print(0 and 5)
  print(True and 3)
  print(False and 3)
  """result
  5
  0
  0
  3
  False"""
  # or의 단축평가(short-circuit evaluation), and의 반대
  print(3 or 5)
  print(3 or 0)
  print(0 or 5)
  print(True or 3)
  print(False or 3)
  """result
  3
  3
  5
  True
  3"""
  ```

- **복합 연산자**

  복합 연산자는 연산과 대입이 함께 진행

  가장 많이 활용되는 경우는 반복문을 통해서 개수를 카운트하거나 할 때

  | 연산자  | 내용       |
  | ------- | ---------- |
  | a += b  | a = a + b  |
  | a -= b  | a = a - b  |
  | a *= b  | a = a * b  |
  | a /= b  | a = a / b  |
  | a //= b | a = a // b |
  | a %= b  | a = a % b  |
  | a **= b | a = a ** b |

- **기타 연산자**

  - Concatenation

    숫자가 아닌 자료형은 `+` 연산자를 통해 합칠 수 있다.

  - Containment Test

    `in` 연산자를 통해 속해있는지 여부를 확인할 수 있다.

  - Identity

    `is` 연산자를 통해 동일한 object인지 확인할 수 있다. 

  - Indexing/Slicing

    `[]`를 통한 값 접근 및 `[:]`을 통한 슬라이싱 

    ```python
      # tuple, list 덧셈
      # dictionary 불가
      print([1, 2, 3] + [4, 5, 6])
      print((1, 2, 3) + (4, 5, 6))
      
      # 기본 정수인 -5부터 256까지의 id는 언제나 동일
      # 해당 기본 정수들은 자주 사용되는 정수로써, 이미 메모리 공간안에 띄워두고 사용
      print(id(-5))
      print(id('hi'))
      print(id('hi') is id('hi'))
      print(id('-5') is id('-5'))
      """result
      1835230208
      1835230208
      64406048
      64406048
      False
      False"""
      
      a = 3
      b = 3
      print(a is b)
      """result : True"""
      
      c = 257
      d = 257
      print(c is d)
      """result : False"""
    ```
  
- **연산자 우선순위**(PEMDAS)

  : 외우지 말고 그냥 괄호 사용

  1. `()`을 통한 grouping
  2. Slicing
  3. Indexing
  4. 제곱연산자 **
  5. 단항연산자 +, - (음수/양수 부호)
  6. 산술연산자 *, /, %
  7. 산술연산자 +, -
  8. 비교연산자, `in`, `is`
  9. `not`
  10. `and`
  11. `or`

  ```python
  # 제곱 연산자가 단항 연산자보다 고순위
  -3 ** 4
  """result : -81"""
  ```

---

- **기초 형변환**(Type Conversion, Typecasting)

  파이썬은 자유롭게 데이터타입 변환 가능

- **암시적 변환**(Implicit Type Conversion)

  사용자가 의도하지 않았지만, 파이썬 내부적으로 자동으로 형변환 하는 경우

  - bool
  - Numbers (int, float, complex)

  ```python
  # boolean + integer
  print(False + 3)
  print(True + 3)
  ```

- **명시적 형변환**, Explicit Type Conversion

  ```python
  str(1) + '등'
  """result : 1등"""
  
  b = '3.5'
  float(b) + 5
  """result : 8.5"""
  
  # ASCII code
  ord('a')
  """result : 97"""
  
  hex(97)
  hex(35)
  int('0x23' + 'ab', 16)
  """result : 9131"""
  ```

---

## Sequence 자료형

- 데이터의 순서대로 나열된 형식

  (정렬되었다라는 뜻은 아니다)

- 파이썬에서의 기본적인 시퀀스 타입

  1. 리스트(list)
  2. 튜플(tuple)
  3. 레인지(range)
  4. 문자열(string)
  5. 바이너리(binary)

  - **list**

    ```python
    [value1, value2, value3]
    ```

    리스트는 대괄호`[]` 를 통해 만들 수 있습니다.

    값에 대한 접근은 `list[i]`를 통해 합니다.

    ```python
    # 선언
    l = []
    ll = list()
    ```

  - **tuple**

    ```python
    (value1, value2)
    ```

    - 튜플은 리스트와 유사하지만, `()`로 묶어서 표현

    - **수정 불가능(immutable)**하고, **읽기만 가능**

    - 직접 사용하는 것보다는 파이썬 내부에서 사용

    - tuple끼리 `+` 가능

    - tuple끼리 복사하면 처음에는 같은 주소값을 갖으나,

      복사된 tuple에 요소를 추가하는 등의 조작을 하면 다른 주소의 다른 객체로 분리

      `Binding_python-tutor.png` 참고

    ```python
    x = 3
    y = 5
    x, y = y, x
    print(x, y)
    """result : 5"""
    # x, y라고 하는 tuple literal이 바뀐 것이며, tuple 바뀐게 아니다.
    ```
    
    ```python
    tp = (1, 2, 3, 4, 5)
      tp2 = 1, 2, 3, 4, 5
    print(tp)
      print(tp2)
    """result
      (1, 2, 3, 4, 5)
    (1, 2, 3, 4, 5)"""
    ```

  - **range()**

    숫자의 시퀀스를 나타내기 위해 사용

    - 기본형 : `range(n)`

  	  0부터 n-1까지 값을 가짐

    - 범위 지정 : `range(n, m)`

      n부터 m-1까지의  값

    - 범위 및 스텝 지정 : `range(n, m, s)`

      n부터 m-1까지 +s만큼 증가

    ```python
  range_ex = range(0,6)
  list_ex = [0, 1, 2, 3, 4, 5]
  print(range_ex)
  print(list_ex)
  print(type(range_ex))
  print(type(list(range_ex)) is type(list_ex))
  """result
  range(0, 6)
    [0, 1, 2, 3, 4, 5]
  <class 'range'>
    True"""
    ```
  
  print(list(range(0, -9, -1)))
  """result : [0, -1, -2, -3, -4, -5, -6, -7, -8]"""
  
  ```
  
  ```
  
- 시퀀스에 활용할 수 있는 연산자/함수
  
    | operation    | 설명                    |
    | ------------ | ----------------------- |
    | `x in s`     | containment test        |
    | `x not in s` | containment test        |
    | `s1 + s2`    | concatenation           |
  | `s * n`      | n번만큼 반복하여 더하기 |
  | `s[i]`       | indexing                |
  | `s[i:j]`     | slicing                 |
  | `s[i:j:k]`   | k간격으로 slicing       |
  | `len(s)`     | 길이                    |
  | `min(s)`     | 최솟값                  |
  | `max(s)`     | 최댓값                  |
  | `s.count(x)` | x의 개수                |
  
    ```python
    # 숫자 0이 6개 있는 list
    [0] * 6
    """result
    [0, 0, 0, 0, 0, 0]"""
    
    # [:]
    location = ['서울', '대구', '대전', '부산', '평창', '광주']
    print(location[2:-1])
    print(location[::-1])
    """result
    ['대전', '부산', '평창']
    ['광주', '평창', '부산', '대전', '대구', '서울']"""
    
    pal = 'racecar'
    pal == pal[::-1]
  """result
  True"""
  
    l = [1, 2, 2, 2, 3]
  l.count(2)
    """result
  3"""
    ```
  
- **set**, **dictionary**
  
  세트는 수학에서의 집합과 동일하게 처리 
  
  세트는 중괄호`{}`를 통해 만들며, 순서가 없고 **중복된 값이 없다.**
  
  `set(list)`는 복잡도가 꽤 크다.
  
    ```python
  {value1, value2, value3}
    ```
  
  | 연산자/함수         | 설명   |
  | ------------------- | ------ |
  | a - b               | 차집합 |
  | a \| b `(| : pipe)` | 합집합 |
  | a & b               | 교집합 |
  | a.difference(b)     | 차집합 |
  | a.union(b)          | 합집합 |
  | a.intersection(b)   | 교집합 |
  
    ```python
    # 차, 합, 교집합
    set_a = {1, 2, 3}
    set_b = {3, 6, 9}
    print(set_a - set_b)
    print(set_a.difference(set_b))
    """result
    {1, 2}
  {1, 2}"""
  
    # 중복제거
    l = [1, 2, 2, 2, 3]
    l = list(set(l))
    print(l)
    """result
  [1, 2, 3]"""
    ```
  
- **dictionary**
  
    ```python
  {Key1:Value1, Key2:Value2, Key3:Value3, ...}
  ```
  
    - 딕셔너리는 `key`와 `value`가 쌍으로 이뤄져있으며, 궁극의 자료구조입니다.
    - `{}`를 통해 만들며, `dict()`로 만들 수도 있습니다.
    
  - `key`는 immutable한 모든 것이 가능하다. (불변값 : string, integer, float, boolean, tuple, range)
  - `value`는 `list`, `dictionary`를 포함한 모든 것이 가능하다.
  
    ```python
    # 선언 방법
    dic_a = {}
    dic_b = dict()
    
    # 전화번호부 만들기
    # python 3.6 미만 버전은 dictionary를 print해보면 계속해서 순서가 달라짐
    # python 3.6 부터는 정해진 규칙에 따라 일정하게 출력
    dic_a['서울'] = '02'
    dic_a['경기'] = '031'
    dic_a['인천'] = '032'
    dic_b = {'서울':'02', '경기':'031', '인천':'032'}
    print(dic_a)
    print(dic_b)
    """result
    {'서울': '02', '경기': '031', '인천': '032'}
    {'서울': '02', '경기': '031', '인천': '032'}"""
    
    # key, value
    # list와는 다르므로, list()를 선언해줘야 list와 완벽히 같아짐
    lk = dic_a.keys()
    lv = dic_a.values()
    print(lk)
    print(lv)
    """result
    dict_keys(['서울', '경기', '인천'])
    dict_values(['02', '031', '032'])"""
    ```

---

## 정리

- mutable : 이미 선언되어 있는, 인덱싱 되어있는 값을 지정하여 수정할 수 있다.

  immutable : 이미 선언되어 있는, 인덱싱이 완료된 값을 지정하여 수정할 수 없다.

- **Ordered**, Sequence

  - `'String'` : immutable
  - `[list]` : mutable
  - `(tuple)` : immutable
  - `range()` : immutable
- **Unordered**

  - `{set}` : mutable
  - `{dictionary}` : mutable