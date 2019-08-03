# Core Exercises

---

Gathered exercises which I could learn sth critical.

예제 및 문제를 풀던 중 깨달음을 주었던 문제 및 풀이들을 모았습니다.

---

## 자료형 다루기, Handliing Data Types

1. Floating point rounding error

   파이썬에서 float는 실수를 표현하는 과정에서 같은 값으로 일치되지 않는다.

   아래의 a, b 값이 같은지 확인하기 위한 코드를 작성하라.

   ```python
   a = 0.1 * 3
   b = 0.3
   ```

2. 별로 직사각형 쉽게 만들기

   가로의 길이가 n, 세로의 길이가 m인 직사각형 형태를 출력하라.

   ```python
   n = 5
   m = 9
   ```

3. 다음 중 mutable / immutable 자료형을 구분하시오.

   ```
   String List Tuple Range Set Dictionary
   ```

4. 다음은 여러 사람의 혈액형(A, B, AB, O)에 대한 데이터이다.

   반복문을 사용하여 key는 혈액형의 종류, value는 인원 수인 딕셔너리를 만들고 출력하시오.

   ```python
   blood_types = ['A', 'B', 'A', 'O', 'AB', 'AB', 'O', 'A', 'B', 'O', 'B', 'AB']
   ```

5. Palindrome은 앞에서부터 읽었을 때와 뒤에서부터 읽었을 때 같은 단어를 뜻한다.

   따라서, ‘a’ ‘nan’ ’토마토’ 모두 palindrome에 해당한다.

   Palindrome일 시 `True`, 아닐 시 `False`를 반환하는 함수를 작성하라.

---

## 함수 & 클래스, Function & Class

1. 재귀함수 사용의 장점 및 단점을 작성하라.

2. 다음 주어지는 class에 대한 세가지 문제의 해답을 작성하시오.

   ```python
   class Calculator:
     count = 0
     
     def info(self):
       print('계산기')
       
     @staticmethod
     def add(a, b):
       Calculator.count == 1
       print(f'{a} + {b}는 {a + b}이다.')
       
     @classmethod
     def history(cls):
       print(f'총 {cls.count}번 계산')
   ```

   1. Instance Method, Static Method, Class Method에 해당하는 함수는 각각 무엇인가?
   2. 각각의 함수(메서드)를 실행하는 코드를 작성하라.
   3. 파라미터 `self`와 `cls`에는 어떤 값이 할당되는가?

3. 다음 함수의 return 값이 얼마인지 계산하시오.

   ```python
   def arguments(a=3, b=4, *args, **kwargs):
       result = a + b
       for arg in args:
           if arg > 2:
               result += arg
       for v in kwargs.values():
           if v > 2:
               result += v
       return result
   
   print(arguments(1, 6, 1, 2, 3, 4, c=2, d=10))
   ```

4. 다음과 같은 2개의 class를 선언한 후,

   사각형의 넓이를 구하는 `get_area()` 메소드를 구현하고,

   `print(instance object)`를 통해 정사각형인지 확인할 수있도록 하라.

   1. `class Point`

      instance attribute로 x, y 좌표값을 갖고 있다.

   2. `class Square`

      instance attribute로 두 개의 점을 갖고 있으며, 아래의 두 메소드를 갖고 있다.

      - 두 점을 `좌측상단점/우측하단점` 형태로 사용하여 넓이를 구하는 `get_area()` 메소드

      - 두 점을 활용하여 사각형을 만들었을 때, 이 사각형이 정사각형인지 확인할 수 있는 `__repr__` 메소드

   ```python
   p1 = Point(3, 4)
   p2 = Point(6, 8)
   s1 = Square(p1, p2)
   print(s1.get_area())
   print(s1)
   ```

   