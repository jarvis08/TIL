## Built-in Functions

OS X 한글 Encoding 오류

`#-\*- coding:utf-8 -\*-`



## is* 시리즈

- `isinstance(object, class/tuple)`

  object가 class/tuple의 instance인지 판별
  
- `set_1.issubset(set_2)` : `set_1`이 `set_2`의 부분집합인지 판별

- `isalpha(string)` : string의 내용이 알파벳인지 판별

- `isdigit(string)` : string의 내용이 숫자인지 판별



## String

- `string.replace('변경 전', '변경 후')` : string의 일부를 원하는 글자로 수정

  ```python
  print('touch example_{}.txt'.format(i))
  print(f'touch example_{i}.txt')
  print('touch example'+ str(i) + '.txt')
  ```
  
- `string.isdigit()` : 문자열이 양의 정수로만 구성되어 있는지 판별

  ```python
  # input()은 모든 입력을 string으로 반화
  num = input()
  if num.isdigit():
      print('숫자입니다.')
  else:
      print('문자입니다.')
  ```

- `string.isalpha()` : 문자열의 내용이 알파벳으로만 구성되어 있는지 판별

- `.strftime()` : 날짜/시간 format에서 날짜/시간 요소만 추출

  `.isoformat()` : %를 사용하지 않으며, 날짜/시간 format을 유지한 채 자료형만 string으로 변환

  ```python
  # datetime.datetime 날짜와 시간이 함께 포함되어 있으므로 date 함수 사용
  from datetime import date, timedelta
  
  yesterday = date.today() - timedelta(days=1)
  print(yesterday)
  yesterday = yesterday.strftime('20%y%m%d')
  print(yesterday)
  """result
  2019-07-16
  20190716"""
  ```

  ```python
  # isoformat은 %를 사용하지 않으며, 날짜/시간 format을 유지한 채 자료형만 string으로 변환
  from datetime import date
  
  yesterday = (date.today() - timedelta(days=1)).isoformat().replace('-', '')
  ```



## List

### Sort

- `list.sort()` : 원본을 오름차순으로 정렬
- `list.sort(reverse=True)` : 원본을 내림차순으로 정렬
- `sorted(list)` : 오름차순한 결과를 반환



### Reverse

- `list.reverse()` : 원본의 순서를 역으로 변환
- `reversed(list)` : 역순으로 변환한 list를 반환
- `list.count(element)` : list 내의 element 개수를 반환
- `cmp(list_1, list2)` : 두 list가 같은 요소들 만을 포함하는지 확인



### Insert/Delete Element

- `list.append()`  : 요소를 뒤에 삽입
- `list.extend()` : 리스트를 연장



### Insert 속도 비교

1. `list.append()`

   가장 느린 속도

2. 인덱스 지정하여 변경

   `list.append()`보다 훨씬 빠른 속도

   ```python
   l = [0] * 10
   for i in range(1, 11):
       l[i] = i
   ```

3. `range` 사용하기

   세 방법 중 가장 빠른 속도이나 사용할 수 있는 상황이 매우 제한적

   ```python
   l = range(1, 11)
   ```

   

## Dictionary

- `dic.keys()` : key 값들의 나열을 dictionary의 특수한 자료형으로 나타내어 반환

- `dic.values()` : value 값들의 나열을 dictionary의 특수한 자료형으로 나타내어 반환

- `dic.items()` : (value, key) 값들의 나열을 dictionary의 특수한 자료형으로 나타내어 반환

- `dict.get(key)` : dict[key] 의 결과값을 반환하며, 없을 시 error가 아닌 None을 반환

  ```python
  dictionary = {'first' : {'second' : 3}}
  print(dictionary.get('first').get('second'))
  """result
  3"""
  ```



## Set

- `set( list )` : list를 집합으로 전환

  ```python
  count = len(set(winner) & set(ur_lotto))
  # winner list와 ur_lotto list를 비교할 때
  # for문을 이용하는 것 보다 빠른 속도로 같은 요소의 개수를 구함
  ```
  
  
  

---

### file 'r / w / a'

- `with open('파일명', '파일 조작 유형', encoding='utf-8') as f:`
- `f.readline()` : 한 줄 읽기
- `f.readlines()` : 모든 문장 읽기
- `f.write()` : 한 번 쓰기
- `f.writelines()` : 모두 쓰기
- 파일 조작 3가지
  - `'r'`: read
  - `'w'`: write
  - `'a'`: append

### filter

- `filter(function, iterable)`

  function에 의해 처리되는 각각의 요소에 대해 Boolean 값을 반환

- 결과값

  - `True` : 유지
  - `False` : 제거

  ``` python
  foo = [2, 9, 27, 3, 4, 5]
  print(list(filter(lambda x: x % 3 == 0, foo)))
  """result
  [9, 27, 3]"""
  ```

### map( )

```python
# a = input()
# b = a.split(" ")
# num1 = int(b[0])
# num2 = int(b[1])

## 위의 네 줄을 생략할 수 있는 map()
# iterable 한 tuple 형태를 응용

# 8 3 이라고 input을 받아서 사칙연산을 수행하는 예제
num1, num2 = map(int, input().split(" "))
print(num1 + num2)
print(num1 - num2)
print(num1 * num2)
print(num1 / num2)

# iterable 하다면 모두 가능하므로 list로 변형해도 가능
num1, num2 = list(num1, num2 = map(int, input().split(" ")))
```

---

## Special Method, Magic Method

- `__method__` 와 같은 형태로 underscore가 두개씩 양 옆에 붙은 method를 의미
- Built-in  Method이며, 기본적으로 python에서 제공

```python
from myPackage.math.formula import pi
dir(pi)
"""result
['__annotations__',
 '__call__',
 '__class__',
 '__closure__',
 '__code__',
 '__defaults__',
 '__delattr__',
 '__dict__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__get__',
 '__getattribute__',
 '__globals__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__kwdefaults__',
 '__le__',
 '__lt__',
 '__module__',
 '__name__',
 '__ne__',
 '__new__',
 '__qualname__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__sizeof__',
 '__str__',
 '__subclasshook__']"""
```