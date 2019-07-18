# SSAFY_Week3_Day4

 **참고자료** : ./50_SSAFY/8ython/notes/04.data_structure.jpynb

---

## 문자열 메소드 활용

- `Duck.run()`

  - `Duck` : 주어 혹은 목적어
  - `run` : 동사
  - `()` : '주어'가 '동사' 하는 방법, **'method'**

- 범위

  - Destructive Method

    원본을 변형시키며 조작, 대체로 return 값이 없다.

    `sort()`, `.reverse()`

  - Non-destructive Method

    원본을 유지하며, 대체로 return값을 통해 전달 받은 변경 내용을 다른 변수에 저장해야 유지 가능

    `.capitalize()`

  ---

### 변형

```python
# dir()을 통해 어떤 메소드를 사용할 수 있는지 확인 가능
dir('3')
```

- `.capitalize()` : 맨 앞글자를 대문자로, 나머지를 소문자로 변환

- `.title()` : 어포스트로피(`'`)나 공백 이후를 대문자로 변환

- `.upper()` : 모두 대문자로 변환

- `lower()` : 모두 소문자로 변환

- `swapcase()` : 대 ↔ 소문자로 변경하여 반환

- **`delimeter.join(iterable)`**

  Iterable 을 해당 문자열을 separator 로 합쳐서 문자열로 반환

  ```python
  # join
  a = '가나 다라'.split(' ')
  print(a)
  b = ' '.join(a)
  print(b)
  """result
  ['가나', '다라']
  가나 다라"""
  ```

- `.replace(old, new[, count])`[¶](http://localhost:8888/notebooks/notes/04.data_structures.ipynb#.replace(old,-new[,-count]))

  바꿀 대상 글자를 새로운 글자로 바꿔서 반환

  count를 지정하면 해당 갯수만큼만 시행

  ```python
  a = '가나 다라'
  b = a.replace('가나', '차카')
  print(b)
  """result
  차카 다라"""
  ```

- `strip([chars])`, 글씨 제거 

  특정한 문자들을 지정하면, 양쪽을 제거하거나 왼쪽을 제거하거나(lstrip) 오른쪽을 제거(rstrip)

  지정하지 않으면 공백을 제거

  ```python
  # '\n'과 같은 공백 문자 제거에 자주 사용
  d = 'This is strip.\n'
  print(d)
  print('--------')
  print(d.strip())
  print('--------')
  print(d.lstrip())
  print('--------')
  print(d.rstrip())
  print('--------')
  
  """result
  This is strip.
  
  --------
  This is strip.
  --------
  This is strip.
  
  --------
  This is strip.
  --------"""
  ```

  ---

### 탐색 및 검증

- `.find(x)` : 첫 번째 x 요소의 위치를 반환, 없으면 -1 반환

- `.index(x)` : 첫 번째 x 요소의 위치를 반환, 없으면 오류

- `split()` : 문자열을 특정한 단위로 나누어 리스트로 반환

- 다양한 확인 메소드 : 참/거짓 반환

  ```python
  # .isalpha : alphabet인지 확인
  .isalpha(), .isdecimal(), .isdigit(), .isnumeric(), .isspace(), .issuper(), .istitle(), .islower()
  dir('string')
  ```

---

## 리스트 메소드 활용

### 값의 추가 및 삭제, 모두 desctructive method(mutable 자료를)

- `.append(x)` : 리스트에 값 추가

  ```python
  caffe = ['bana', 'star']
  caffe.append('been')
  caffe += ['bene']
  # 주의
  caffe += 'bene'
  print(caffe)
  
  """result
  ['bana', 'star', 'been', 'bene', 'b', 'e', 'n', 'e']"""
  ```

- `.extend(iterable)` : 리스트에 **iterable**(list, range, tuple, string*유의*) 값을 추가

  **list에 list를 붙일 때 사용!**

  ```python
  caffe = ['bana', 'star']
  caffe.extend(['hollys'])
  caffe += ['bene']
  # 주의
  caffe.extend('been')
  caffe += 'ediya'
  print(caffe)
  """result
  ['bana', 'star', 'hollys', 'bene', 'b', 'e', 'e', 'n', 'e', 'd', 'i', 'y', 'a']"""
  ```

- 어렵게 넣기

  ```python
  caffe[len(caffe):] = ['ediya']
  ```

- `insert(i, x)` : index i에 x를 추가

  길이를 넘어서는 index 지정 시, 무조건 맨 마지막에 추가

  ```python
  caffe = ['bana', 'w']
  caffe.insert(0, 'star')
  caffe.insert(-1, 'bene')
  caffe.insert(len(caffe), 'hollys')
  print(caffe)
  """result
  ['star', 'bana', 'bene', 'w', 'hollys']"""
  ```

- `remove(x)` : 리스트에서 값이 x인 첫 번째 요소를 삭제

  없는 요소 선언 시 에러

  ```python
  numbers = [1, 2, 3, 4, 1, 2, 3, 1, 2]
  while 1 in numbers:
      numbers.remove(1)
  print(numbers)
  ```

- `pop(i)`

  정해진 위치 `i`에 있는 값을 삭제하며, 그 항목을 반환

  `i`가 지정되지 않으면 마지막 항목을 삭제하고 반환

  ```python
  numbers = [1, 2, 3, 4, 5, 6]
  print(numbers.pop())
  print(numbers.pop(0))
  print(numbers)
  
  """result
  6
  1
  [2, 3, 4, 5]"""
  ```

- 실행 속도

  `remove()`는 하나씩 모두 확인

  `pop()`은 index로 바로 search

  대체로 index를 받는 method가 빠른 속도로 실행

  ---

### 탐색 및 정렬

- `.index(x)` : x값을 찾아 index 반환

  x 값이 없을 시 에러

- `.count(x)` : x값을 가진 요소의 개수 반환

  ```python
  # remove는 무조건 하나만 삭제
  # 따라서 몇 회 반복할 지 정해줘야 할 때 count() 사용 가능
  def delete_all(l, target):
      for i in range(l.count(target)):
          l.remove(target)
          
  numbers = [1, 2, 3, 4, 1, 2, 3, 1, 2]
  delete_all(numbers, 1)
  print(1 in numbers)
  """result
  False"""
  ```

- `.sort()` : 원본 list를 정렬하고 None 반환

  `.sorted()` : 원본을 변형시키지 않으며, sort된 list를 반환

  ```python
  nums = [1, 3, 2]
  nums.sort()
  print(nums)
  # 내림차순
  nums.sort(reverse=True)
  print(nums)
  """result
  [1, 2, 3]
  [3, 2, 1]"""
  ```

  

- `.reverse()` : 원본 list를 역순으로 변형시키고 None 반환

  `.reversed()` : 원본을 변형시키지 않으며, reverse된 list를 반환

  ```python
  classroom = [1, 2, 3]
  l = reversed(classroom)
  print(reversed(classroom))
  print(list(l))
  print(classroom)
  print(classroom.reverse())
  print(classroom)
  """result
  <list_reverseiterator object at 0x05A16FB0>
  [3, 2, 1]
  [1, 2, 3]
  None
  [3, 2, 1]"""
  ```

  ---

### 데이터 복사, Copying Data

- **Binding**

  : 데이터가 저장된 주소와 변수명(호출명)을 연결하는 작업

  ```python
  # 3 값을 저장하고 있는 공간의 주소를 num이라는 이름으로 연결(binding)
  num = 3
  
  # 모든 변수/함수 등은 주소값을 가지고 있다.
  def hello():
      pass
  print(hello)
  """result
  <function hello at 0x059FD4F8>"""
  ```

  - `python tutor`를 이용하면 시각화해서 확인 가능

  - 파이썬에서 모든 변수는 **객체의 주소**를 가지고 있을 뿐이다.

    `Binding_python-tutor.png`  참고

  - 위와 같이 변수를 생성하면 객체를 생성하고, 변수에는 객체의 주소를 저장

  - 변경가능한(mutable) 자료형과 변경불가능한(immutable) 자료형은 서로 다르게 동작

    - `python tutor`를 이용하면 시각화해서 확인 가능

      `Binding_python-tutor.png` 참고

      ```python
      # list_1과 list_2는 서로 다른 주소의 객체
      list_1 = [1, 2, 3]
      list_2 = list[:]
      # list_2 = [1, 2, 3]
      # list_2 = list(list_1)
      
      # list_1과 list_3는 같은 주소의 객체를 공유
      list_3 = list_1
      # list_3를 변경하면 list_1도 변경
      list_3.append(4)
      
      # tuple_1과 tuple_2는 같은 주소의 객체를 공유
      tuple_1 = (1, 2, 3)
      tuple_2 = tuple_1
      # tuple_2를 변경 시 tuple_1과 다른 주소의 객체로 분리
      tuple_2 += (2, 2)
      ```

- 복사 방법

  - 얕은 복사, Shallow Copy

    Shallow Copy는 1차원 까지만 가능

    ```python
    # list
    a = [1]
    b = a
    b = list(a)
    
    # dictionary
    c = {'a':b}
    d = dict(c)
    ```

    - **2차원 객체**

      기본적으로 2차원 객체는 1차원 객체 안에 2차원 객체의 주소를 저장

      `Binding_matrix.png` 참조

    - **2차원 객체 복사**

      2차원 객체들의 주소를 담은 1차원 객체만을 다른 주소의 다른 객체로 복사하며,

      2차원 객체의 주소를 공유

      ```python
      # matrix_2는 2차원의 요소까지 복사 불가
      # 1차원 list를 다른 주소의 객체로 생성하지만,
      # 2차원 요소를 불러올 때에는 matrix_1의 2차원 객체로 접근
      # matrix_1의 2차원 요소 값을 변경해도 matrix_1의 원본 값이 변경
      matrix_1 = [[1, 2], [3, 4]]
      matrix_2 = matrix_1[:]
      matrix_2[0][0] = 3
      ```

      `Binding_matrix_shallow-copy.png` 참고

      - Tuple의 경우 복사 후 수정을 가하면, 다른 주소의 객체로 분리 후 수정

  - 깊은 복사, Deep Copy

    ```python
    import copy
    matrix_1 = [[1, 2], [3, 4]]
    matrix_2 = copy.deepcopy(matrix_1)
    print(matrix_2[0][0])
    ```

    `Binding_matrix_deep-copy.png` 참고

- `.clear()` : 리스트의 모든 항목 삭제

  ---

### List Comprehension

- 간단하게 list 만들기

  ```python
  # cubic 계산
  numbers = range(1, 11)
  cubic_list = [num ** 3 for num in numbers]
  print(cubic_list)
  ```

- 쌍으로 묶기

  ```python
  girls = ['jane', 'iu', 'mary']
  boys = ['justin', 'david', 'kim']
  pairs = [(girl, boy) for girl in girls for boy in boys]
  """result
  [('jane', 'justin'), ('jane', 'david'), ('jane', 'kim'), ('iu', 'justin'), ('iu', 'david'), ('iu', 'kim'), ('mary', 'justin'), ('mary', 'david'), ('mary', 'kim')]"""
  ```

- x < y < z < 50 조건을 만족하는 피타고라스 방정식의 해 찾기

  ```python
  # 일반적인 반복문
  goras =[]
  for x in range(50):
      for y in range(x+1, 50):
          for z in range(y+1,50):
              if x**2 + y**2 == z**2:
                  goras.append((x, y, z))
  print(goras)
  
  # List Comprehension
  goras = [(x,y,z) for x in range(50) for y in range(x+1, 50)
           for z in range(y+1, 50) if x**2 + y**2 == z**2]
  print(goras)
  
  ```

- 문장 모음 제거

  ```python
  words = 'Life is too short, you need python!'
  jaum = ''.join([word for word in words if word not in 'aeoiuy'])
  print(jaum)
  """result
  Lf s t shrt,  nd pthn!"""
  ```

---

## 딕셔너리 메소드 활용

### 추가 및 삭제

- `.pop(key[, default])`

  key가 딕셔너리에 있으면 제거하고 그 값을 반환하며, 없다면 default를 반환

  default가 없는 상태에서 딕셔너리에 없으면 KeyError가 발생

  ```python
  my_dict = {'apple': '사과', 'banana': '바나나'}
  print(my_dict.pop('apple'))
  print(my_dict.pop('apple', 'No Key'))
  print(my_dict)
  사과
  No Key
  {'banana': '바나나'}
  ```

- `.update(**kwargs)`

  값을 제공하는 key, value로 덮어씀

  ```python
  # .updaye(**kwargs)
  # **kwargs는 unpacking 된 형태 or dic 형태로 인자 전달
  my_dict = {'apple': '사과', 'banana': '바나나', 'melon': '멜론'}
  
  # Unpacking 형태
  my_dict.update(apple='포도', banana='감자')
  print(my_dict)
  """result
  {'apple': '포도', 'banana': '감자', 'melon': '멜론'}"""
  
  # Dictionary 형태
  update_dict = {'apple':'사과', 'banana':'바나나'}
  my_dict.update(update_dict)
  print(my_dict)
  
  """result
  {'apple': '사과', 'banana': '바나나', 'melon': '멜론'}"""
  ```

- `.get(key[, default])`

  key를 통해 value를 탐색

  default 값인 `None`으로 인해 절대 KeyError가 발생하지 않음

  ```python
  my_dict = {'apple': '사과', 'banana': '바나나', 'melon': '멜론'}
  
  # my_dict['pineapple'] 오류
  print(my_dict.get('pineapple'))
  ```

  ---

### Dictionary Comprehension

- 그냥 `{ }` 안에 넣기만 하면 set으로 적용

  ```python
  cubic_dict = {x for x in range(1,11)}
  type(cubic_dict)
  """result
  set"""
  ```

- `:` 사용하여 dic으로 인지

  ```python
  cubic_dict = {x:x**3 for x in range(1,11)}
  type(cubic_dict)
  print(cubic_dict)
  """result
  <class 'dict'>
  {1: 1, 2: 8, 3: 27, 4: 64, 5: 125, 6: 216, 7: 343, 8: 512, 9: 729, 10: 1000}"""
  ```

- 예제

  - 미세먼지 hell 찾기

    ```python
    dusts = {'서울': 72, '대전': 82, '구미': 29, '광주': 45, '중국': 200}
    hell = {region:dusts[region] for region in dusts if dusts[region] > 80}
    hell = {k : v for k, v in dusts.items() if v > 80}
    print(hell)
    """result
    {'대전': 82, '중국': 200}"""
    ```

  - `else` 사용하기

    Value에 따라 '나쁨' , '보통' 으로 변경하기

    `else`를 쓰려면 `저장형태 - 조건문 - 반복문`의 순서로 구성

    ```python
    # 미세먼지 농도가 80초과는 나쁨 80이하는 보통으로 하는 value를 가지도록
    hell_yea = {k:'나쁨' if v > 80 else '보통' for k,v in dusts.items()}
    print(hell_yea)
    """result
    {'서울': '보통', '대전': '나쁨', '구미': '보통', '광주': '보통', '중국': '나쁨'}"""
    ```

  - `elif` = `else + if`사용하기

    ```python
    dusts = {'서울': 72, '대전': 82, '구미': 29, '광주': 45, '중국': 200}
    else_if = {k:'매우나쁨' if v > 120 else '나쁨' if v > 80 else '보통' if v > 30 else '좋음' for k,v in dusts.items()}
    print(else_if)
    """result
    {'서울': '보통', '대전': '나쁨', '구미': '좋음', '광주': '보통', '중국': '매우나쁨'}"""
    ```

---

## Set 메소드 활용

### 추가 및 삭제

- Data 조작, CRUD

  `Create`

  `Read`

  `Update`

  `Delete`

- `.add(elem)` : elem을 set에 추가

  ```python
  fruits = {"사과", "바나나", "수박"}
  fruits.add('복숭아')
  print(fruits)
  {'사과', '복숭아', '수박', '바나나'}
  ```

- `.update(*others)` : 여러가지의 값을 추가

  반드시  `list`와 같은 iterable한 값

  ```python
  fruits = {"사과", "바나나", "수박"}
  fruits.update(['복숭아', '포도', '복숭아', '포도', '복숭아', '포도', '복숭아', '포도'])
  print(fruits)
  """result
  {'복숭아', '포도', '수박', '바나나', '사과'}"""
  ```

- `.remove(elem)`

  elem을 세트에서 삭제하고, 없으면 KeyError 발생

- `.discard(elem)`

  x를 세트에서 삭제하고, 없어도 에러가 발생하지 않는다.

- `.pop()`

  **임의의 원소**를 제거해 반환

  지정하여 제거 불가

- `map()`, `zip()`, `filter()`
  - `map(function, iterable)`
    - Iterable의 모든 원소에 function을 적용한 후 그 결과를 돌려줍니다.
    - 대표적으로 iterable한 타입 - list, dict, set, str, bytes, tuple, range
    - return은 map_object 형태로 됩니다.
    - function은 사용자 정의 함수도 가능합니다.
  - `zip(*iterables)`
    - 복수 iterable한 것들을 모아준다.
    - 결과는 튜플의 모음으로 구성된 zip object를 반환한다.
    - 아래와 같이 사용가능하다.
    - zip은 반드시 길이가 같을 때 사용해야한다. 가장 짧은 것을 기준으로 구성한다.
    - 길이가 긴 것을 맞춰서 할 수도 있지만, 사용할 일이 없다.
  - `filter(function, iterable)`
    - iterable에서 function의 반환된 결과가 참인 것들만 구성하여 반환한다.

### Set Comprehension

- List Comprehension과 유사

  ```python
  # 책의 단어별 counting
  import requests
  url = 'http://composingprograms.com/shakespeare.txt'
  novel = requests.get(url).text
  # print(novel)
  # 위 print는 너무 데이터 길이가 너무 길어서 에러
  """result
  IOPub data rate exceeded."""
  
  # 일반적인 개수 구하기
  words = novel.split(' ')
  print(len(words))
  print(len(set(words)))
  """result
  906255
  41450"""
  
  ## Set Comprehension 사용하기
  print(len({word for word in words}))
  # 6글자이며 pelindrome인 단어 찾기
  print(len({word for word in words if len(word) >= 6 and word == word[::-1]}))
  """result
  41450
  33624
  1"""
  ```

---

## 함수형 언어의 잠재

- `map(fucntion, iterable)`

  함수 또한 주소 값을 가지고 있는 객체이며, 이를 직접 사용하는 것이 가능

  ```python
  def add_3(x):
      return x+3
  nothing = add_3
  print(nothing(3))
  """result
  6"""
  ```

  - Iterable한 객체의 모든 원소에 function을 적용한 후 반환

    ```python
    def add_3(x):
        return x+3
    l = [1, 2, 3]
    print(map(add_3, l))
    print(list(map(add_3, l)))
    """result
    <map object at 0x09316BF0>
    [4, 5, 6]"""
    
    numbers = [1, 2, 3]
    str_numbers = [str(num) for num in numbers]
    print(str_numbers)
    print(list(map(str, numbers)))
    """result
    ['1', '2', '3']
    ['1', '2', '3']"""
    ```

  - 대표적으로 iterable한 타입 - `list`, `dict`, `set`, `str`, `bytes`, `tuple`, `range`

  - `return`은 **`map_object 형태`**로 됩니다.

  ---

- `zip(*iterables)`

  - 복수 iterable한 것들을 모아서 묶음
  - tuple의 모음으로 구성된 zip object를 반환

  ```python
  # 남여 묶기
  girls = ['jane', 'iu', 'mary']
  boys = ['justin', 'david', 'kim']
  zipped = zip(girls, boys)
  print(zipped)
  print(list(zipped))
  print(dict(zipped))
  """python
  <zip object at 0x097B3878>
  [('jane', 'justin'), ('iu', 'david'), ('mary', 'kim')]
  {'jane': 'justin', 'iu': 'david', 'mary': 'kim'}"""
  ```

  ```python
  a = '123'
  b = '567'
  for digit_a, digit_b in zip(a, b):
      print(digit_a, digit_b)
  
  """result
  1 5
  2 6
  3 7"""
  ```

  - 짝이 맞는 값들 까지만 계산

    ```python
    num1 = [1, 2, 3]
    num2 = ['1', '2']
    list(zip(num1, num2))
    """result
    [(1, '1'), (2, '2')]"""
    ```

  - `itertools` 라이브러리의 `zip_logest` 메소드를 사용하면 없는 값에 대해 `fillvalue` 값을 부여하여 대체 가능

    ```python
    from itertools import zip_longest
    list(zip_longest(num1, num2, fillvalue=0))
    """result
    [(1, '1'), (2, '2'), (3, 0)]"""
    ```

  ---

- `filter(function, iterable)`

  iterable에서 function의 반환된 **결과가 참인 것들만 구성하여 반환**

  ```python
  def even(n):
      return not n%2
  
  numbers = range(1, 20)
  list(filter(even, numbers))
  
  """result
  [2, 4, 6, 8, 10, 12, 14, 16, 18]"""
  ```

  

- `함수의 인자(parameter)` : 함수를 선언할 때 설정한 값
  `인수(argument)` : 함수를 호출할 때 넘겨주는 값