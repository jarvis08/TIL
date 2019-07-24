# SSAFY_Week4_Day2

 **참고자료** : ./50_SSAFY/8ython/notes/07.OOP_basic.ipynb

---

## 예외처리

- `try`, `except`

  - `try`절을 실행
  - 예외가 발생되지 않으면, `except`없이 실행이 종료
  - 실행 중도에 지정한 예외 사항 발생시, 남은 부분을 수행하지 않고 `except` 실행

  ```python
  try:
      codeblock1
  # 예외 내용은 변수의 특정 값일 수 있으며,
  # Error 내용으로 지정할 수 있다
  # i.g., ValueError
  except 예외 내용:
      codeblock2
  ```

  ---

- 복수 예외 처리

  두 가지 예외를 모두 처리 가능

  ```python
  try:
      codeblock1
  except (예외1, 예외2):
      codeblock2
  ```

  ```python
  # 예시) 100을 사용자가 입력한 값으로 나눈 후 출력하는 코드
  try:
      var = input('숫자를 입력')
      print(100/int(var))
  except ValueError:
      print('바보야 숫자 입력하라니까')
  except ZeroDivisionError:
      print('0이 아닌 숫자를 입력하세요.')
  except:
      print('뭘 이상하게 했는지는 모르곘지만, 에러 났음')
  ```

  - **에러는 순차적으로 수행되므로, 가장 작은 범주부터 시작해야 한다!**

    ```python
    try:
        var = input('숫자를 입력 : ')
        print(100/int(var))
    except Exception:
        print('뭘 이상하게 했는지는 모르곘지만, 에러 났음')
    except ValueError:
        print('바보야 숫자 입력하라니까')
    except ZeroDivisionError:
        print('0이 아닌 숫자를 입력하세요.')
    
    """result
    숫자를 입력 : 0
    뭘 이상하게 했는지는 모르곘지만, 에러 났음"""
    # ZeroDivisionError이지만, 앞에서 걸러져버림
    ```

  ---

- 에러 문구 처리

  에러 문구를 print, 전달 할 수 있다.

  ```python
  try:
      num_list = [1, 2, 3]
      print(num_list[4,4])
  except Exception as err:
      print(err)
  """result
  list indices must be integers or slices, not tuple"""
  ```

  ---

- `else`

  에러가 발생하지 않는 경우 수행

  ```python
  try:
      codeblock1
  except 예외:
      codeblock2
  else:
      codeblock3
  ```

  ```python
  try:
      num_list = [1, 2, 3]
      num_list[2]
  except:
      print('에러 났어요')
  else:
      print(num_list[2])
  """result
  3"""
  ```

  ---

- `finally`

  예외 발생 여부에 상관 없이, 반드시 수행해야하는 문장

  ```python
  try:
      codeblock1
  except 예외:
      codeblock2
  finally:
      codeblock3
  ```

  ```python
  try:
      students = {'john':'cs', 'jaeseok':'math'}
      students['minji']
  except KeyError as em:
      print(f'{em} 는 딕셔너리에 없는 키입니다.')
  finally:
      print('곧 쉬는 시간. 조금만 힘내')
  
  """result
  'minji' 는 딕셔너리에 없는 키입니다.
  곧 쉬는 시간. 조금만 힘내"""
  ```

  ---

- `raise`

  예외 발생시키기

  ```python
  # ValueError와 같은 에러 유형도 함수이다!
  # 따라서 ('메세지')를 부여하여 에러와 함께 특정 메세지 출력 가능
  num = input('숫자 :: ')
  if num.isalpha():
      raise ValueError('숫자를 입력하세요')
  ```

  ---

- `assert`

  보통 상태를 검증하는데 사용되며, 무조건 `AssertionError` 발생

  ```python
  assert 'Boolean expression', 'error message'
  ```

  'Boolean expression'의 검증식이 거짓일 경우 `AssertionError` 발생

  `raise`는 **항상 예외**를 발생시키며, `assert`는 **지정 예외**만 발생한다는 점에서 차이

  ```python
  def my_div(num1, num2):
      assert type(num1) == int and type(num2) == int, '정수가 아닙니다.'
      try:
          result = num1 / num2
      except ZeroDivisionError as err:
          print(f'{err} 오류가 발생했습니다.')
      else:
          return result
  my_div(4,2)
  """result
  2.0"""
  ```

  _보통 예외 처리 용이 아닌, TDD 방법론과 같이 Test를 할 때 사용_

---

 **참고자료** : ./50_SSAFY/8ython/notes/07.OOP_basic.ipynb

## 객체 지향 프로그래밍, Object-Oriented Programming( OOP)

- 절차적(Procedual) 프로그래밍

  순서도(Flow Chart)를 그릴 수 있는 프로그래밍

  프로그램 규모가 커지면, 유지 보수가 어려움

- 객체 지향 프로그래밍

  - 이론으로만 존재했던 OOP를 smalltalk에서 처음으로 구현

  - (언어학+철학)에서 시작된 개념

    세상을 있는 그대로 표현할 수 있도록 사람처럼 묘사하는,

    서술적인 사고 방식(주어 + 동사)을 구현하고자 시작

  - (주어 + 동사)는 (Object + Predicate)의 개념으로 발전

  - 따라서 Object를 '객체' 보다는 '사물'로 해석하는 것이 이해하기 쉬우며,

    _객체 지향 프로그래밍 = 사실적인 프로그래밍_ 이라고 해석할 수 있다.

  ```python
  # Object + Predicate
  # 'hello' + islower()
  # 'hello'를 islower() 해라
  'hello'.islower()
  ```

<wikipedia - 객체지향 프로그래밍>

> 객체 지향 프로그래밍(영어: Object-Oriented Programming, OOP)은 컴퓨터 프로그래밍의 패러다임의 하나이다. 객체 지향 프로그래밍은 컴퓨터 프로그램을 명령어의 목록으로 보는 시각에서 벗어나 여러 개의 독립된 단위, 즉 "객체"들의 모임으로 파악하고자 하는 것이다. 각각의 객체는 메시지를 주고받고, 데이터를 처리할 수 있다.
>
> 명령형 프로그래밍인 절차지향 프로그래밍에서 발전된 형태를 나타내며, 기본 구성요소는 다음과 같다.

- 클래스, Class

  - 같은 종류(또는 문제 해결을 위한)의 집단에 속하는 **속성(attribute)**과 **행위(behavior)**를 정의한 것으로, 객체지향 프로그램의 기본적인 사용자 정의 데이터형(user define data type)이라고 할 수 있다.

  - class는 프로그래머가 아니지만 해결해야 할 문제가 속하는 영역에 종사하는 사람이라면 사용할 수 있고, 다른 클래스 또는 외부 요소와 독립적으로 디자인하여야 한다.

    ---

    우리는 처음 나무라는 식물을 배울 때 위키피디아로 배우지 않는다.

    이것도, 그것도, 저것도 나무라는 것을 학습해가며 나무들의 **공통점** 및 **체계**를 익힌다.

    OOP에서는 그 **분류 체계**를 **`class`**라고 칭하며, Object들을 **구조화** 한다.

  ---

- 인스턴스, Instance

  - class의 instance/object(실제로 메모리상에 할당된 것)

  - object는 자신 고유의 속성(attribute)을 가지며 class에서 정의한 행위(behavior)를 수행

  - object의 행위는 class에 정의된 행위에 대한 정의(method)를 공유함으로써 메모리를 경제적으로 사용

  - `instanciate` : instance를 만드는 작업

    ---

    나무 `class`에 속하는 여러가지 나무들을 `instance`라고 한다.

  ---

- 속성, Attribute

  - 클래스/인스턴스 가 가지고 있는 속성(값)

    ---

    `class` 나무

    `instance` 바오밥 나무

    `attribute` 나무 재질

    ```python
    # 복소수의 실수, 허수 속성
    complex_num = 3 + 2j
    print(complex_num.real)
    print(complex_num.imag)
    
    """result
    3.0
    2.0"""
    ```

  ---

- 메서드, Method

  - 클래스/인스턴스 가 할 수 있는 행위(함수)

    ```python
    complex_num = 3 + 2j
    print(complex_num.conjugate)
    print(complex_num.conjugate())
    """result
    <built-in method conjugate of complex object at 0x05649EF0>
    (3-2j)"""
    ```

  - `dir(객체)` : 사용 가능한 method 확인

    ```python
    nums = [3, 2, 1]
    dir(nums)
    """result
    ['__add__',
     '__class__',
     '__contains__',
     '__delattr__',
     '__delitem__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getitem__',
     '__gt__',
     '__hash__',
     '__iadd__',
     '__imul__',
     '__init__',
     '__init_subclass__',
     '__iter__',
     '__le__',
     '__len__',
     '__lt__',
     '__mul__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__reversed__',
     '__rmul__',
     '__setattr__',
     '__setitem__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     'append',
     'clear',
     'copy',
     'count',
     'extend',
     'index',
     'insert',
     'pop',
     'remove',
     'reverse',
     'sort']"""
    ```

  ---

| class / type | instance                 | attributes       | methods                                |
| ------------ | ------------------------ | ---------------- | -------------------------------------- |
| `str`        | `''`, `'hello'`, `'123'` | _                | `.capitalize()`, `.join()`, `.split()` |
| `list`       | `[]`, `['a', 'b']`       | _                | `.append()`, `reverse()`, `sort()`     |
| `dict`       | `{}`, `{'key': 'value'}` | _                | `.keys()`, `.values()`, `.items().`    |
| `int`        | `0`, `1`, `2`            | `.real`, `.imag` | `.conjugate()`                         |



## 클래스 및 인스턴스

- class 정의하기 (클래스 객체 생성하기)

  - class 정의 시 항상 첫글자는 대문자로

    ```python
    class ClassName:
    ```

  - **선언과 동시에 클래스 객체 생성**
  - 또한, 선언된 공간은 local scope로 사용
  - `멤버 변수` : class 내에서 정의된 attribute 변수
  - `method` : class 내에서 정의된 함수(`def`)

  ```python
  class Person:
      name = 'John'
      
      def sleep():
          print('쿨쿨')
  
  john = Person()
  john.sleep()
  """result
  TypeError: sleep() takes 0 positional arguments but 1 was given"""
  # class의 method에는 자동으로 자기 자신을 의미하는 self 인자 전달
  # 따라서 argument 초과로 TypeError 발생
  ```

- 인스턴스 생성하기

  - instance object는 `ClassName()`을 호출하여 선언

    ```python
    class Person:
        name = ''
    p1 = Person()
    print(isinstance(p1, Person))
    """result
    True"""
    ```

  - instance object와 class object는 **서로 다른 namespace를 소유**

  - **인스턴스 => 클래스 => 전역 순으로 탐색**

  ```python
  class TestClass:
      name = 'Test Class'
  tc = TestClass()
  print(type(tc))
  dir(tc)
  
  tc.name = 'tc'
  print(tc.name)
  
  """result
  <class '__main__.TestClass'>
  tc"""
  ```

  ```python
  class Phone:
      power = False
      number = ''
      model = 'Samsung Galaxy S10'
      
      def on(self):
          if not self.power:
              self.power = True
              print('----------------')
              print(f'{self.model}')
              print('----------------')
      def off(self):
          if self.power:
              self.power = False
              print('전원이 꺼집니다.')
  
  my_phone = Phone()
  my_phone.on()
  my_phone.model = 'iPhone_X'
  print(my_phone.model)
  print(isinstance(my_phone, Phone))
  print(my_phone)
  """result
  ----------------
  Samsung Galaxy S10
  ----------------
  'iPhone_X'
  True
  <__main__.Phone object at 0x00B4F430>
  """
  ```

  - `__repr__`, `__str__`

    특정 object를 `print()` 할 때 보이는 값인 `print(my_phone)`과

    object 자체가 보여주는 값인  `my phone`도 변형시킬 수 있다.

    - 차이점

      `__str__`은 꼭 `print()` 안에 넣지 않고도 사용 가능

    ```python
    # print(my_phone) 출력 결과를 이쁘게 만들기
    # class phone 안에 다음과 같은 method 추가
    class Phone:
        def __str__(self):
            return self.model
    
    print(my_phone)
    # 아래 코드는 __repr__이 아닌 __str__로 작성했기에 가능
    my_phone
    """result
    iPhone_X
    iPhone_X"""
    ```

  ---

  ```python
  # 예제) list 객체 만들기
  class MyList:
      data = []
      def append(self, element):
          ## 다음 세 가지 방법 모두 가능
          # self.data.append(element)
          # self.data.extend([element])
          self.data += element
      
      def pop(self):
          # return self.data.pop()
          last = self.data[-1]
          # self.data = self.data[:-1]
          del self.data[-1]
          return last
      
      def reverse(self):
          self.data = self.data[::-1]
          
      def count(self, element):
          # self.data.count(x)
          cnt = 0
          for el in self.data:
              if el == element:
                  cnt += 1
          return cnt
      
      def clear(self):
          self.data = []
      
      def __repr__(self):
          """객체가 표현될 때 쓰이는 문자열"""
          return f'내 리스트에는 {self.data} 이 담겨있다.'  
      
      
      def __str__(self):
          return f'내 리스트에는 {self.data} 이 담겨있다.'
  
  ml = MyList()
  print(ml)
  ml.append(2)
  ml.pop()
  ml.append(4)
  ml.append(3)
  print(ml)
  ml.reverse()
  print(ml)
  print(ml.count(3))
  ml.clear()
  print(ml)
  
  """result
  내 리스트에는 [] 이 담겨있다.
  내 리스트에는 [4, 3] 이 담겨있다.
  내 리스트에는 [3, 4] 이 담겨있다.
  1"""
  ```

  ---

- 용어 정리

  ```python
  class Person:                     #=> 클래스 정의(선언) : 클래스 객체 생성
      name = 'unknown'              #=> 멤버 변수(data attribute)
      def greeting(self):           #=> 멤버 메서드(메서드)
          return f'{self.name}' 
  richard = Person()      # 인스턴스 객체 생성
  tim = Person()          # 인스턴스 객체 생성
  tim.name                # 데이터 어트리뷰트 호출
  tim.greeting()          # 메서드 호출
  ```

  ---

- `self` : instance object 자기자신

  - C++ 혹은 자바에서의 `this` 키워드와 동일하지만,

    `this`는 method 선언시 parameter로 넣지 않음

  - 특별한 상황을 제외하고는 **무조건 메서드에서 `self`를 첫번째 인자로 설정**

  - __`self` 작성 이유 : method는 instance object가 함수의 첫번째 인자로 전달__

  - `self`는 예약어가 아니기 때문에 대신 다른 용어를 사용해도 작동,

    하지만 관례상, 무조건 `self`라는 용어를 사용

  ---

- class와 instance의 **namespace**

  - class를 정의하면, class object가 생성되고 해당되는 namespace 생성

  - instance를 만들게 되면, instance object가 생성되고 해당되는 namespace 생성

  - instance의 특정 attribute에 접근 시 **instance => class 순으로 탐색**

  - **immutable한 attribute를 변경 시, 변경된 데이터를 instance object의 namespace에 저장**

  - mutable한 attribute의 경우 class object namespace를 변경하기 때문에

    instance object 별로 `__init__` 생성자를 통해 따로 namespace를 분리하지 않는 한,

    모든 instance object가 같은 namespace의 attribute를 공유

    ```python
    # mutable한 attribute의 namespace 공유
    class Person:
        num = []
        def ad(self, number):
            self.num.append(number)
    
    p1 = Person()
    print(p1.num)
    p1.ad(3)
    p2 = Person()
    print(p2.num)
    """result
    []
    [3]"""
    ```

    ```python
    # immutable한 attribute 예시
    class Person:
        name = '동빈'
    
    p1 = Person()
    # 현재 p1의 name은 class object namespace에서 '동빈'을 가져옴
    
    p1.name = '조동빈'
    # instance object의 attribute를 변경하였기 때문에
    # p1의 name attribute는 instance object의 namespace에 '조동빈'이라고 저장됨
    
    p2 = Person()
    print(p1.name)
    print(p2.name)
    
    """result
    조동빈
    동빈"""
    ```

    ```python
    # mutable class object attribute 수정하기
    class Person:
        num_people = 0
        name = ' '
        
        def __init__(self, name):
            # class명.attribute 호출 시, instance에서 class namespace 접근 가능
            Person.num_people += 1
            self.num_people += 1
            self.name = name
    
    p1 = Person('조동빈')
    print(p1.num_people)
    print(p1.name)
    print(Person.num_people)
    print(Person.name)
    """result
    2
    조동빈
    1
     """
    ```

- 즉, instance object를 생성했더라도

  생성자 혹은 method를 이용해 attribute를 변경하지 않으면

  attribute는 class object의 attribute namespace를 사용

  - python은 `private`, `public`과 같은 class object namespace 구분이 없기 때문에 발생하는 현상

  - python 또한 `private`, `public`과 같은 존재가 있지만, 관례상 미사용

  - 생성자인 `__init__`을 이용하여 instance object 생성 시 초기값 부여 가능

    class object로부터 attribute를 분리시키는 기능을 수행

  ---

- 생성자 / 소멸자

  - Special Method, Magic Method

    `__init__`, `__str__` 와 같이 양쪽에 underscore가 존재하는 method

  - 생성자, Constructor

    instance object가 생성될 때 호출되는 함수

    - class object로부터 attribute의 namespace를 분리시킬 수 있다.

      i.g., `self.name = '조동빈'`

    - `class_name.attribute_name = 값` 형태를 통해 class object의 namespace 또한 접근 가능

      i.g., `Person.num_people += 1`

  - 소멸자

    instance object가 소멸되는 과정에서 호출되는 함수

  ```python
  def __init__(self):
      print('생성될 때 자동으로 호출되는 메서드입니다.')
  
  def __del__(self):
      print('소멸될 때 자동으로 호출되는 메서드입니다.')
  ```

  ```python
  class Person:
      num_people = 0
      name = ' '
      
      def __init__(self, name):
          Person.num_people += 1
          self.num_people += 1
          self.name = name
  
  p1 = Person('조동빈')
  print(p1.num_people)
  print(p1.name)
  print(Person.num_people)
  print(Person.name)
  """result
  2
  조동빈
  1
   """
  ```