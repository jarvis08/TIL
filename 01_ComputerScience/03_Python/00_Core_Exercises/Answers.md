# Answers

---

## 자료형 다루기, Handliing Data Types

1. Floating point rounding error

  ```python
  import sys
  
  a = 0.1*3
  b = 0.3
  print(abs(a-b) <= sys.float_info.epsilon)
  ```

  
  ```python
    import math

    a = 0.1*3
  b = 0.3

    print(math.isclose(a,b))
  ```

  ```python
    a = 0.1*3
    b = 0.3

  print(abs(a-b) <= 0.000000001)
    print(abs(a-b) <= 1e-10)
  ```

2. Making square with `*`

   - 방법 1

     ```python
     n = 5
     m = 9
     
     print((('*' * n) + '\n') * m)
     ```

   - 방법 2

     ```python
     for m in range(9):
         for n in range(5):
             print('*', end='')
         print()
     ```

3. Mutable / Immutable data types

   immutable한 자료형 또한 추가 및 삭제가 가능하지만, 기존의 값을 변경시킬 수는 없다.

   ```
   # Mutable
   List, Set, Dictionary
   
   # Immutable
   String, Tuple, Range
   ```

4. Counting values

   - `collections` 라이브러리 사용

     ```python
     import collections
     
     blood_types = ['A', 'B', 'A', 'O', 'AB', 'AB', 'O', 'A', 'B', 'O', 'B', 'AB']
     print(collections.Counter(blood_types))
     ```

   - `set` 자료형과  `list.count()` 메소드 사용

     ```python
     blood_dic = dict()
     for blood in set(blodd_types):
         blood_dic[blood] = blood_types.count(blood)
     print(blood_dic)
     ```

   - `dict.get()`

     ```python
     blood_dic = dict()
     for person in blood_types:
         if not blood_dic.get(person):
             blood_dic[person] = 0
         blood_dic[person] += 1
     print(blood_dic)
     ```

5. Palindrome

   ```python
   def palindrome(word):
       return word == word[::-1]
   ```

---

## 함수 & 클래스, Function & Class

1. 재귀함수 사용의 장점 및 단점

   ```
   - 장점
   	1. 코드가 비교적 간결
   	2. 직관적이어서 가독성이 좋다.
   	3. 반복되는 문제를 구조화 하기 좋다.
   	
   - 단점 : 복잡도가 증가하여 비효율적
   ```

2. Class Method의 종류

   - Instance Method

     인스턴스 객체의 attribute를 조작하는 데에 사용

   - Static Method

     주로 class/instance object의 attribute와 무관한 계산 등의 활동에 사용

   - Class Method

     클래스 객체의 attribute를 조작하는 데에 사용

   ```
   instance method :: info()
   static mathod :: add()
   class method :: history()
   ```

   ```python
   # Instance Object 생성
   calc = Calculator()
   # Instance Method 사용
   calc.info()
   
   # Static Method 사용
   Calculator.add(a, b)
   
   # Class Method 사용
   class :: Calculator.history()
   ```

   ```
   self :: instance object
   cls :: class object
   ```

3. Parameters

   - Position Argument
   - Keyword Argument
   - 가변인자 list

   ```python
   print(arguments(1, 6, 1, 2, 3, 4, c=2, d=10))
   # 1, 6은 Position Argument로써 a, b에 대입
   # 1, 2, 3, 4는 가변인자이자 tuple type으로 *args에 대입
   # c=2와 d=10은 Keyword Argument인 **kwargs에 dictionary type으로 대입
   정답 : 24
   ```

4. Getting argument with another class's instance object

   ```python
   #-*-coding: utf-8
   class Point:
       x = 0
       y = 0
   
       def __init__(self, x, y):
           self.x = x
           self.y = y
   
   
   class Square:
       def __init__(self, point_1, point_2):
           self.p1 = point_1.x , point_1.y
           self.p2 = point_2.x, point_2.y
       
       def get_area(self):
           return abs(self.p1[0] - self.p2[0]) * abs(self.p1[1] - self.p2[1])
       
       def __repr__(self):
           isit = False
           if abs(self.p1[0] - self.p2[0]) == abs(self.p1[1] - self.p2[1]):
               isit = True
           return 'p1과 p2를 이용한 사각형은 정사각형인가? {}'.format(isit)
   
   p1 = Point(3, 4)
   p2 = Point(6, 8)
   s1 = Square(p1, p2)
   print(s1.get_area())
   print(s1)
   """result
   12
   p1과 p2를 이용한 사각형은 정사각형인가? False"""
   ```

   
