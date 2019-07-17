# Python Standard Libraries

---

## String

- `string.replace('변경 전', '변경 후')` : string의 일부를 원하는 글자로 수정

  ```python
  print('touch example_{}.txt'.format(i))
  print(f'touch example_{i}.txt')
  print('touch example'+ str(i) + '.txt')
  ```

## strftime, isoformat

- 날짜/시간을 string으로 변환

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
  # isoformat은 %를 사용하지 않아도 되므로
  from datetime import date
  
  yesterday = (date.today() - timedelta(days=1)).isoformat().replace('-', '')
  ```

  

---

## List

`list.reverse()` : list 원본의 순서를 역으로 변환

---

## Dictionary

`list(dic.keys())` : dictionary의 key 값들을 list로 변환

`dic.get(key).get(key)` : 해당 key의 value를 반환, 없을 시 error 대신 None 반환

---

## Set

: set은 중복 요소가 없으며, 오름차순

`set( list )` : list를 집합으로 전환

```python
count = len(set(winner) & set(ur_lotto))
# winner list와 ur_lotto list를 비교할 때
# for문을 이용하는 것 보다 빠른 속도로 같은 요소의 개수를 구함
```

---

## Sort

`sorted([ ])` : list를 오름차순으로 sort

---

## collections

- 내장 함수이며, 알고리즘 속도 개선을 위해 자주 사용
- dictionary와 유사하지만, 다른 type

```python
import collections
blood_types = ['A', 'B', 'A', 'O', 'AB', 'AB', 'O', 'A', 'B', 'O', 'B', 'AB']
print(collections.Counter(blood_types))
"""python
Counter({'A': 3, 'B': 3, 'O': 3, 'AB': 3})"""
```

------

## File r / w / a

`with open('파일명', '파일 조작 유형', encoding='utf-8') as f:`

`f.readlines()` : 모든 문장 읽기

`f.readline()` : 한 줄 읽기

`f.write()` : 한 번 쓰기

`f.writelines()` : 모두 쓰기

- 파일 조작 3가지

  `'r'`: read

  `'w'`: write

  `'a'`: append

---

## os

`os.listdir()`: 현재 디렉토리 내부의 모든 파일, 디렉토리를 리스트에 저장

`os.rename(현재 파일명, 바꿀 파일명)`: 파일명 변경

`os.system()`: Terminal에서 사용하는 명령어 사용

```shell
os.system('touch example.txt')
# example.txt 파일 생성
os.system('rm example.txt')
# example.txt 파일 제거
```

`os.chdir()`: 작업 폴더를 현 위치에서 해당 위치로 옮김

---

## Random

`random.sample( [], int )` : [ ] 중 int 개 만큼 비복원 추출

`random.choice( [ ] )` : [ ] 중 1개를 임의 복원 추출

---

## map( )

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

## math

- `sqrt()` : 루트
- `floor()` : 버림
- `ceil()` : 올림