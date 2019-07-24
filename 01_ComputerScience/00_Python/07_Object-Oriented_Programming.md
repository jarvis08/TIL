# Object-Oriented Programming

  **참고자료** : ./50_SSAFY/8ython/notes/08.OOP_advanced.ipynb

---

## 클래스 변수 / 인스턴스 변수

- 클래스 변수
  - 클래스의 속성입니다.
  - 클래스 선언 블록 최상단에 위치합니다.
  - `Class.class_variable` 과 같이 접근/할당합니다.

  ```python
  class TestClass:
      class_variable = '클래스변수'
      ...
      
  TestClass.class_variable  # '클래스변수'
  TestClass.class_variable = 'class variable'
  TestClass.class_variable  # 'class variable'
  
  tc = TestClass()
  tc.class_variable  # 인스턴스 => 클래스 => 전역 순서로 네임스페이스를 탐색하기 때문에, 접근하게 됩니다.
  ```

- 인스턴스 변수
  - 인스턴스의 속성입니다.
  - 메서드 정의에서 `self.instance_variable` 로 접근/할당합니다.
  - 인스턴스가 생성된 이후 `instance.instance_variable` 로 접근/할당합니다.

  ```python
  class TestClass:
      def __init__(self, arg1, arg2):
          self.instance_var1 = arg1
          self.instance_var2 = arg2
      
      def status(self):
          return self.instance_var1, self.instance_var2   
      
  tc = TestClass(1, 2)
  tc.instance_var1  # 1
  tc.instance_var2  # 2
  tc.status()  # (1, 2)
  ```

  ------

  ```python
  class TestClass:
      class_variable = '클래스변수'
      ...
      
  TestClass.class_variable  # '클래스변수'
  TestClass.class_variable = 'class variable'
  print(TestClass.class_variable)  # 'class variable'
  """result
  class variable"""
  TestClass.class_variable = 'ClaaaaSS variable'
  print(TestClass.class_variable)
  """result
  ClaaaaSS variable"""
  tc = TestClass()
  print(tc.class_variable)
  """result
  ClaaaaSS variable"""
  tc.class_variable = '인스턴스가 변경한 인스턴스의 class_variable'
  print(tc.class_variable)
  print(TestClass.class_variable)
  """result
  인스턴스가 변경한 인스턴스의 class_variable
  ClaaaaSS variable"""
  ```

  ------

## 인스턴스 메서드 / 클래스 메서드 / 스태틱(정적) 메서드

- 인스턴스 메서드
  - 인스턴스가 사용할 메서드 입니다.
  - **정의 위에 어떠한 데코레이터도 없으면, 자동으로 인스턴스 메서드가 됩니다.**
  - **첫 번째 인자로 `self` 를 받도록 정의합니다. 이 때, 자동으로 인스턴스 객체가 `self` 가 됩니다.**

  ```python
    class MyClass:
        def instance_method_name(self, arg1, arg2, ...):
            ...
  
    my_instance = MyClass()
    my_instance.instance_method_name(.., ..)  # 자동으로 첫 번째 인자로 인스턴스(my_instance)가 들어갑니다.
  ```

- 클래스 메서드
  - 클래스가 사용할 메서드 입니다.
  - **정의 위에 `@classmethod` 데코레이터를 사용합니다.**
  - **첫 번째 인자로 `cls` 를 받도록 정의합니다. 이 때, 자동으로 클래스 객체가 `cls` 가 됩니다.**

  ```python
    class MyClass:
        @classmethod
        def class_method_name(cls, arg1, arg2, ...):
            ...
  
    MyClass.class_method_name(.., ..)  # 자동으로 첫 번째 인자로 클래스(MyClass)가 들어갑니다.
  ```

- 스태틱(정적) 메서드
  - 클래스가 사용할 메서드 입니다.
  - **정의 위에 `@staticmethod` 데코레이터를 사용합니다.**
  - **인자 정의는 자유롭게 합니다. 어떠한 인자도 자동으로 넘어가지 않습니다.**

  ```python
    class MyClass:
        @staticmethod
        def static_method_name(arg1, arg2, ...):
            ...
  
    MyClass.static_method_name(.., ..)  # 아무일도 자동으로 일어나지 않습니다.
  ```

  ------

## 정리

```python
class MyClass:
    @classmethod
    def class_method(cls):
        return cls
    
    def instance_method(self):
        return self
    @staticmethod
    def static_method():
        return 'static method'

mc = MyClass()
print(mc.instance_method())
# instance가 class method를 호출해도 class가 호출한 것으로 처리
print(MyClass.class_method())
print(mc.class_method())
# class가 호출하는 것은 맞지만, 자기 자신인 cls를 인수하지 않는다.
# class_method는 자기 자신의 attribute를 처리 할 때 사용
# static_method는 class와 관련 없는 단순 계산 등에 자주 사용
print(MyClass.static_method)
print(mc.static_method)
# class가 instance method를 호출 가능하지만, 사용하지 않는다.
# 대장 내시경 하는데 입으로 관을 넣은 느낌
print(MyClass.instance_method(mc))

"""result
<__main__.MyClass object at 0x057495D0>
<class '__main__.MyClass'>
<class '__main__.MyClass'>
<function MyClass.static_method at 0x05729A98>
<function MyClass.static_method at 0x05729A98>
<__main__.MyClass object at 0x057495D0>"""
```

- instance는 3가지 method 모두에 접근 가능

- 하지만 instance에서 class method와 static method를 호출하지 말 것(가능하다 != 사용한다)

- instance가 할 행동은 모두 instance method로 한정 지어서 설계

- class는, 3가지 method 모두에 접근 가능 하지만,

  class에서 instance method를 호출하지 않도록 한다. (가능하다 != 사용한다)

- class가 할 행동은 다음 원칙에 따라 설계

  - class 자체(`cls`)와 그 속성에 접근할 필요가 있다면 class method로 정의
  - class와 class attribute에 접근할 필요가 없다면 static method로 정의

  ```python
  # 예제
  class Puppy:
      num_dogs = 0
      def __init__(self, name, age):
          self.name = name
          self.age = age
          Puppy.num_dogs += 1
          print(f'강아지 {self.name}을(를) 분양받았습니다.')
      def bark(self):
          print(f'{self.name}, 개가 짖습니다.')
      @classmethod
      def dogs_num(cls):
          print(f'가족이 총 {Puppy.num_dogs} 마리입니다.')
      @staticmethod
      def info():
          print('Puppy Class는 강아지를 만들어주는 Class 입니다.')
  ggamy = Puppy('Ggamy', '?')
  Romeo = Puppy('Romeo', 15)
  janggoon = Puppy('Janggoon', '?')
  ggamy.bark()
  Romeo.bark()
  janggoon.bark()
  Puppy.dogs_num()
  Puppy.info()
  Romeo.info()
  """result
  강아지 Ggamy을(를) 분양받았습니다.
  강아지 Romeo을(를) 분양받았습니다.
  강아지 Janggoon을(를) 분양받았습니다.
  Ggamy, 개가 짖습니다.
  Romeo, 개가 짖습니다.
  Janggoon, 개가 짖습니다.
  가족이 총 9 마리입니다.
  Puppy Class는 강아지를 만들어주는 Class 입니다.
  Puppy Class는 강아지를 만들어주는 Class 입니다."""
  ```

  ------

## 연산자 오버라이딩(중복 정의, 덮어 쓰기)

- 기본적으로 정의된 연산자를 직접적으로 정의하여 활용 가능
- `dir()`을 통해 overriding 할 method 확인

```python
+  __add__   
-  __sub__
*  __mul__
# less than
<  __lt__
# less or equal
<= __le__
== __eq__
!= __ne__
# greater or equal
>= __ge__
>  __gt__
```

- 예제

  연산자 오버라이딩을 이용해 사람의 나이로 비교 가능하게 만들기(>, < 이용)

  ```python
  class Person:
      population = 0
      
      def __init__(self, name, age):
          self.name = name
          self.age = age
          Person.population += 1
          
      def greeting(self):
          print(f'{self.name} 입니다. 반갑습니다!')
          
      def __repr__(self):
           return f'< "name:" {self.name}, "age": {self.age} >'
      
      def __gt__(self, other):
          return self.age >= other.age
  john = Person('john', 34)
  ashley = Person('ashley', 32)
  eric = Person('eric', 30)
  john.greeting()
  ashley.greeting()
  
  people = [john, ashley, eric]
  print(ashley.__gt__(john))
  print(ashley > john)
  ######################################################
  # 인스턴스 객체 별 비교가 가능해졌으므로, sorting도 가능!
  ######################################################
  print(sorted(people))
  """result
  john 입니다. 반갑습니다!
  ashley 입니다. 반갑습니다!
  False
  False
  [< "name:" eric, "age": 30 >, < "name:" ashley, "age": 32 >, < "name:" john, "age": 34 >]"""
  ```

  ------

## 상속, Inheritance

- class에서 가장 큰 특징은 '상속' 기능

- parent class의 모든 attribute, method를 자식 클래스에게 상속하여 코드 재사용성 향상

- (Parent, Child) == (Super, Sub) == (Base, Derived) Class

  어떻게 불러도 상관 없다.

```python
class DerivedClassName(BaseClassName):
    code block
```

- `super()`

  - sub class에 method 추가 구현 가능
  - super class의 내용을 사용하고자 할 때, `super(기존 인자)` 사용

- Method Overriding[¶](http://localhost:8888/notebooks/notes/08.OOP_advanced.ipynb#메서드-오버라이딩)

  메서드를 재정의

- 상속관계에서의 이름공간

  - 기존에 인스턴스 -> 클래스순으로 이름 공간을 탐색해나가는 과정에서 상속관계에 있으면 아래와 같이 확장
  - instance => class => global
  - instance -> 자식 클래스 -> 부모 클래스 -> 전역

- 다중상속

  두개 이상의 클래스를 상속받는 경우이며, 꼬이기 쉬워 잘 사용하지 않는다.

------

- 내가 만든 포켓몬 게임

  ```python
  import random
  
  class Pokemon:
      name = ''
      level = 5
      hp = level * 20
      exp = 0
      
      poke_type = ''
      weak_type = ''
      strong_type = ''
      
      def __init__(self, name):
          self.name = name
      
      def body_striking(self, enemy):
          print(f'\nAttack\t:: {self.name}, {enemy.name}에게 몸통박치기!')
          print(f'{self.name}\t:: 쿵쾅쿵쾅')
          if enemy.hp <= 0:
              print(f'{enemy.name}\t:: 시체 매너좀.')
              return None
          if random.randint(0, 100) > 70:
              print(f'Command\t:: {enemy.name}가 {self.name}의 몸통 박치기를 회피했다!')
          else:
              damage = self.level * 6
              print(f'Command\t:: {enemy.name}은 {self.name}의 몸통 박치기에 의해 {damage} 데미지를 입었다.')
              enemy.hp -= damage
      
      def medicine(self):
          print(f'\nCommand\t:: {self.name}가 회복약을 이용해 hp를 20 회복했다.')
      
      def yell(self):
          print(f'\n{self.name}\t:: 꾸에에에에엑!')
          
      def __str__(self):
          return f'{self.name}\t:: 속성[{self.poke_type}]\t약점[{self.weak_type}]'
  
  class WaterPoke(Pokemon):
      def __init__(self, name):
          self.name = name
          self.poke_type = '물'
          self.weak_type = '번개'
          self.strong_type = '불'
      
      def skill_1(self, enemy):
          if enemy.hp <= 0:
              print(f'{enemy.name}\t:: 시체 매너좀.')
              return None
          print(f'\nAttack\t:: {self.name}, {enemy.name}에게 물대포 발사!')
          print(f'{self.name}\t:: 어푸어푸')
          if self.poke_type == enemy.weak_type:
              damage = self.level * 10
              print(f'Command\t:: {enemy.name}(은)는 {self.name}의 물대포에 {damage} 데미지를 입었다.')
              print(f'Command\t:: {enemy.poke_type} 속성인 {enemy.name}(은)는 추가 데미지를 입었다!')
              enemy.hp -= damage
          elif self.weak_type == enemy.poke_type:
              damage = self.level * 3
              enemy.hp -= damage
              print(f'Command\t:: {enemy.name}(은)는 {self.name}의 물대포에 {damage} 데미지를 입었다.')
              print(f'Command\t:: {enemy.poke_type} 속성인 {enemy.name}에게는 데미지가 감소되었다!')
          else:
              damage = self.level * 5
              print(f'Command\t:: {enemy.poke_type} 속성인 {enemy.name}(은)는 물대포에 {damage} 데미지를 입었다.')
              enemy.hp -= damage
              
  class FirePoke(Pokemon):
      def __init__(self, name):
          self.name = name
          self.poke_type = '불'
          self.weak_type = '물'
          self.strong_type = '번개'
      
      def skill_1(self, enemy):
          print(f'\nAttack\t:: {self.name}, {enemy.name}에게 파이어볼!')
          print(f'{self.name}\t:: 화륵화륵')
          if enemy.hp <= 0:
              print(f'{enemy.name}\t:: 시체 매너좀.')
              return None
          if self.poke_type == enemy.weak_type:
              damage = self.level * 10
              enemy.hp -= damage
              print(f'Command\t:: {enemy.name}(은)는 {self.name}의 파이어볼에 의해 {damage} 데미지를 입었다.')
              print(f'Command\t:: {enemy.poke_type} 속성인 {enemy.name}(은)는 추가 데미지를 입었다!')
          elif self.weak_type == enemy.poke_type:
              damage = self.level * 3
              enemy.hp -= damage
              print(f'Command\t:: {enemy.name}(은)는 {self.name}의 파이어볼에 의해 {damage} 데미지를 입었다.')
              print(f'Command\t:: {enemy.poke_type} 속성인 {enemy.name}에게는 데미지가 감소되었다!')
          else:
              damage = self.level * 5
              enemy.hp -= damage
              print(f'Command\t:: {enemy.poke_type} 속성인 {enemy.name}(은)는 파이어볼에 {damage} 데미지를 입었다.')
  
  class ThunderPoke(Pokemon):
      def __init__(self, name):
          self.name = name
          self.poke_type = '번개'
          self.weak_type = '불'
          self.strong_type = '물'
      
      def skill_1(self, enemy):
          print(f'\nAttack\t:: {self.name}, {enemy.name}에게 백만 볼트!')
          print(f'{self.name}\t:: 지직지직')
          if enemy.hp <= 0:
              print(f'{enemy.name}\t:: 시체 매너좀.')
              return None
          if self.poke_type == enemy.weak_type:
              damage = self.level * 10
              enemy.hp -= damage
              print(f'Command\t:: {enemy.name}(은)는 {self.name}의 백만 볼트에 {damage} 데미지를 입었다.')
              print(f'Command\t:: {enemy.poke_type} 속성인 {enemy.name}(은)는 추가 데미지를 입었다!')
          elif self.weak_type == enemy.poke_type:
              damage = self.level * 3
              enemy.hp -= damage
              print(f'Command\t:: {enemy.name}(은)는 {self.name}의 백만 볼트에 {damage} 데미지를 입었다.')
              print(f'Command\t:: {enemy.poke_type} 속성인 {enemy.name}에게는 데미지가 감소되었다!')
          else:
              damage = self.level * 5
              enemy.hp -= damage
              print(f'Command\t:: {enemy.poke_type} 속성인 {enemy.name}(은)는 백만 볼트에 {damage} 데미지를 입었다.')
  
  
  # 띠리리 매치 중 죽는놈 체크
  def hp_check(poke_1, poke_2):
      if poke_2.hp <= 0:
          print(f'\nVictory\t:: 야생의 {poke_1.name}가 승리했다!')
          return True
  
  # 띠리리 매치 참가 전, 체력 동등하게 맞추기
  def set_ddiriry_hp(heal):
      pika.hp = heal
      pairy.hp = heal
      kobuk.hp = heal
      
  # 띠리리 띠리리 띠리리~
  def ddiriry_ddiriry_ddiririy(poke_1, poke_2):
      """
      0 = 소리지르기
      1 = 몸통 박치기
      2 = 공격 스킬
      3 = 회복약 마시기 스킬"""
      dead = False
      rand = random.choice([0, 1, 2, 3])
      if rand == 0:
          poke_1.yell()
      elif rand == 1:
          poke_1.body_striking(poke_2)
      elif rand == 2:
          poke_1.skill_1(poke_2)
      else:
          poke_1.medicine()
  
      dead = hp_check(poke_1, poke_2)
      if dead:
          print('-------------------------------------------------------전투 종료')
          return None
      return ddiriry_ddiriry_ddiririy(poke_2, poke_1)
  
  
  pairy = FirePoke('파이리')
  kobuk = WaterPoke('꼬부기')
  pika = ThunderPoke('피카츄')
  
  # 1st match
  set_ddiriry_hp(150)
  ddiriry_ddiriry_ddiririy(pairy, kobuk)
  
  # 2nd match
  set_ddiriry_hp(150)
  ddiriry_ddiriry_ddiririy(pairy, pika)
  
  # 3rd match
  set_ddiriry_hp(150)
  ddiriry_ddiriry_ddiririy(pika, kobuk)
  ```

  