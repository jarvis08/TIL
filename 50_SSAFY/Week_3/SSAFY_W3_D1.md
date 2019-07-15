# SSAFY_Week3_Day1

---

# 전체 SSAFY 과정, 배우는 이유

- bootstrap

  css어렵다. 남들이 잘해논거 쓰는게 이득

- django

  객체지향 개념을 가장 잘 공부할 수 있는 framework

- Javascript

  이제 가장 필수적인 언어

  임베디드고 뭐고 모든게 웹으로 얽힘

- Vue.js

  귀요미로 만들어주는 framework

- aws

  전체적인 그림을 그릴 수 있도록 클라우드 경험

---

## Python

### 50_SSAFY/8ython/notes/01.python_intro

`python -i` : git bash에서 python interactive 실행

`.ipynb ` : jupyter notebook 확장자, json 파일 형식으로 저장\

```python
# source image를 center에 정렬하여 표시
<center><img src="./images/01/variable.png", alt="variable"/></center>
```

`PEP-8` : python enhancement proposal, python 개발자들의 표준안

- python keyword

  : `keyword` library에 포함되어 있다. 이 내용들은 함수/클래스/변수명 불가

  https://wikidocs.net/20557

```python
import keyword
keyword.kwlist
```

- docstring : `func.__doc__` 기입 시 docstring 내용의 string 출력

```python
# 주석을 연습해봅시다. 
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

- 기본적으로 파이썬에서는 `;` 을 작성하지 않는다.

  한 줄로 표기할 떄는 `;`를 작성하여 표기할 수 있다.

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

- python tutor

  : VISUALIZE CODE AND GET LIVE HELP

  line by line으로 python 내부에서 어떻게 실행되는지 확인 가능

  http://pythontutor.com/

  http://pythontutor.com/visualize.html#mode=edit

- 자료형은 기본적으로 세가지

  - 숫자
  - 글자
  - 참/거짓(boolean)

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

- complex, 복소수

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

- Boolean

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

- None

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

- String

  - 문자열은 Single quotes(`'`)나 Double quotes(`"`)을 활용하여 표현 가능하다.
  - 단, 문자열을 묶을 때 동일한 문장부호를 활용해야하며, `PEP-8`에서는 **하나의 문장부호를 선택**하여 유지하도록 하고 있습니다. (Pick a rule and Stick to it)
  - 강동주 대표님은 single quote를 주로 사용

  ```python
  pro_said = '김지수 프로님은 말씀하셨다.'
  proo = "오늘은 종례가 없을 거에요."
  print(pro_said, proo)
  print(pro_said + ' ' + proo)
  """result
  김지수 프로님은 말씀하셨다. 오늘은 종례가 없을 거에요.
  김지수 프로님은 말씀하셨다. 오늘은 종례가 없을 거에요.
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

    `PEP-8`에 따르면 이 경우에는 반드시 `"""`를 사용하도록 되어 있다

  ```python
  # excape code를 이용하여 single quote 중복 사용하기
  pro_said = '김지수 프로님은 말씀하셨다.\'오늘은 종례가 없을 거에요.\''
  print(pro_said)
  """result
  김지수 프로님은 말씀하셨다.'오늘은 종례가 없을 거에요.'
  """
  # """ 사용하기
  print("""여러줄에 걸쳐있는 문장은 다음과 같이 표현 가능합니다.
  PEP-8에 따르면 이 경우에는 반드시 이렇게 하도록 합니다.""")
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

  

  문자열을 활용하는 경우 특수문자 혹은 조작을 하기 위하여 사용되는 것으로 `\`를 활용하여 이를 구분한다. 

  | <center>예약문자</center> |   내용(의미)    |
  | :-----------------------: | :-------------: |
  |            \n             |     줄바꿈      |
  |            \t             |       탭        |
  |            \r             |   캐리지리턴    |
  |            \0             |    널(Null)     |
  |           `\\`            |       `\`       |
  |            \'             | 단일인용부호(') |
  |            \"             | 이중인용부호(") |