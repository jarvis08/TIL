# Virtual Function

 컴파일 시에 어떤 함수가 실행될 지 정해지지 않고 런타임 시에 정해지는 일을 가리켜서 **동적 바인딩(dynamic binding)** 이라고 부른다. 동적 바인딩은 가상 함수(virtual function)를 사용했을 때 발생하며, 가상 함수는 `virtual` 키워드를 사용하여 선언한다.

```cpp
#include <iostream>

class Base {

 public:
  Base() { std::cout << "기반 클래스" << std::endl; }
	
  // 가상 함수로 what()을 선언
  virtual void what() { std::cout << "기반 클래스의 what()" << std::endl; }
};

class Derived : public Base {

 public:
  Derived() : Base() { std::cout << "파생 클래스" << std::endl; }
	
  // 가상 함수로 what()을 선언
  void what() { std::cout << "파생 클래스의 what()" << std::endl; }
};

int main() {
  Base p;
  Derived c;
	
  // Base 포인터로 Derived를 가리킴
  Base* p_c = &c;
  // Base 포인터로 Base를 가리킴
  Base* p_p = &p;

  std::cout << " == 실제 객체는 Base == " << std::endl;
  p_p->what();

  std::cout << " == 실제 객체는 Derived == " << std::endl;
  p_c->what();

  return 0;
}
```

```
기반 클래스
기반 클래스
파생 클래스
 == 실제 객체는 Base == 
기반 클래스의 what()
 == 실제 객체는 Derived == 
파생 클래스의 what()
```

만약 `virtual`을 사용하지 않았다면, up casting이 발생하여 `p_p->what()`과 `p_c->what()` 모두 '기반' 이라고 출력됐을 것이다. 하지만, 동적 바인딩을 사용했기 때문에, `p_c`가 `Derived`를 가리키므로 알아서 `Derived`의 `what()`를 실행한다. 이렇게 파생 클래스의 함수가 기반 클래스의 함수를 오버라이드 하기 위해서는 두 함수의 꼴이 정확히 같아야 한다.

 이와 반대로, **정적 바인딩(static binding)**은 컴파일 타임에 어떤 함수가 호출될 지 정해지는 것이다.

<br>

### Override 키워드

함수의 형태가 정확히 같지 않다면 override가 되지 않을 수 있다. 따라서 다음과 같이 override 키워드를 사용한다면, 확실하게 선언해 줄 수 있다.

`void what() override { std::cout << s << std::endl; }`

<br>

### Polymorphism, 다형성

> 이 두 부분은 `employee_list[i]` 가 `Employee` 냐 `Manager` 에 따라서 다르게 동작하게 됩니다. 이렇게 같은 `print_info` 함수를 호출했음에도 불구하고 어떤 경우는 `Employee` 의 것이, 어떤 경우는 `Manager` 의 것이 호출되는 일; 즉 하나의 메소드를 호출했음에도 불구하고 여러가지 다른 작업들을 하는 것을 바로 **다형성(polymorphism)** 이라고 부릅니다.

```cpp
#include <iostream>
#include <string>

class Employee {
 protected:
  std::string name;
  int age;

  std::string position;  // 직책 (이름)
  int rank;              // 순위 (값이 클 수록 높은 순위)

 public:
  Employee(std::string name, int age, std::string position, int rank)
      : name(name), age(age), position(position), rank(rank) {}

  Employee(const Employee& employee) {
    name = employee.name;
    age = employee.age;
    position = employee.position;
    rank = employee.rank;
  }
  Employee() {}
	
  // 가상 함수로 사용하여, 동적으로 바인딩되록 함
  virtual void print_info() {
    std::cout << name << " (" << position << " , " << age << ") ==> "
              << calculate_pay() << "만원" << std::endl;
  }
  // 가상 함수로 사용하여, 동적으로 바인딩되록 함
  virtual int calculate_pay() { return 200 + rank * 50; }
};

class Manager : public Employee {
  int year_of_service;

 public:
  Manager(std::string name, int age, std::string position, int rank, int year_of_service)
      : year_of_service(year_of_service), Employee(name, age, position, rank) {}

  // 가상 함수를 overriding 하도록 선언
  int calculate_pay() override { return 200 + rank * 50 + 5 * year_of_service; }
  void print_info() override {
    std::cout << name << " (" << position << " , " << age << ", "
              << year_of_service << "년차) ==> " << calculate_pay() << "만원"
              << std::endl;
  }
};
class EmployeeList {
  int alloc_employee;        // 할당한 총 직원 수
  int current_employee;      // 현재 직원 수
  Employee** employee_list;  // 직원 데이터

 public:
  EmployeeList(int alloc_employee) : alloc_employee(alloc_employee) {
    employee_list = new Employee*[alloc_employee];
    current_employee = 0;
  }
  void add_employee(Employee* employee) {
    employee_list[current_employee] = employee;
    current_employee++;
  }
  int current_employee_num() { return current_employee; }

  void print_employee_info() {
    int total_pay = 0;
    for (int i = 0; i < current_employee; i++) {
      employee_list[i]->print_info();
      total_pay += employee_list[i]->calculate_pay();
    }

    std::cout << "총 비용 : " << total_pay << "만원 " << std::endl;
  }
  ~EmployeeList() {
    for (int i = 0; i < current_employee; i++) {
      delete employee_list[i];
    }
    delete[] employee_list;
  }
};
int main() {
  EmployeeList emp_list(10);
  emp_list.add_employee(new Employee("노홍철", 34, "평사원", 1));
  emp_list.add_employee(new Employee("하하", 34, "평사원", 1));

  emp_list.add_employee(new Manager("유재석", 41, "부장", 7, 12));
  emp_list.add_employee(new Manager("정준하", 43, "과장", 4, 15));
  emp_list.add_employee(new Manager("박명수", 43, "차장", 5, 13));
  emp_list.add_employee(new Employee("정형돈", 36, "대리", 2));
  emp_list.add_employee(new Employee("길", 36, "인턴", -2));
  emp_list.print_employee_info();
  return 0;
}
```

<br>

### 가상 함수의 Destructor는 가상 함수로 선언!

> `delete p` 를 하더라도, `p` 가 가리키는 것은 `Parent` 객체가 아닌 `Child` 객체 이기 때문에, 위에서 보통의 `Child` 객체가 소멸되는 것과 같은 순서로 생성자와 소멸자들이 호출되어야만 합니다. 그런데 실제로는, `Child` 소멸자가 호출되지 않습니다.
>
> 소멸자가 호출되지 않는다면 여러가지 문제가 생길 수 있습니다. 예를 들어서, `Child` 객체에서 메모리를 동적으로 할당하고 소멸자에서 해제하는데, 소멸자가 호출 안됬다면 **메모리 누수(memory leak)**가 생기겠지요.
>
> 하지만 `virtual` 키워드를 배운 이상 여러분은 무엇을 해야 하는지 알고 계실 것입니다. 단순히 `Parent` 의 소멸자를 `virtual` 로 만들어버리면 됩니다. `Parent` 의 소멸자를 `virtual` 로 만들면, `p` 가 소멸자를 호출할 때, `Child` 의 소멸자를 성공적으로 호출할 수 있게 됩니다.

```cpp
#include <iostream>

class Parent {
 public:
  Parent() { std::cout << "Parent 생성자 호출" << std::endl; }
  virtual ~Parent() { std::cout << "Parent 소멸자 호출" << std::endl; }
};
class Child : public Parent {
 public:
  Child() : Parent() { std::cout << "Child 생성자 호출" << std::endl; }
  ~Child() { std::cout << "Child 소멸자 호출" << std::endl; }
};
int main() {
  std::cout << "--- 평범한 Child 만들었을 때 ---" << std::endl;
  { 
    // 이 {} 를 빠져나가면 c 가 소멸된다.
    Child c; 
  }
  std::cout << "--- Parent 포인터로 Child 가리켰을 때 ---" << std::endl;
  {
    Parent *p = new Child();
    delete p;
  }
}
```

<br>

### 가상 함수의 구현 원리

>  간혹 **가상** 이라는 이름 때문에 혼동하시는 분이 계시는데, `virtual` 키워드를 붙여서 가상 함수로 만들었다 해도 실제로 존재하는 함수이고 정상적으로 호출도 할 수 있습니다. 또한 모든 함수들을 디폴트로 가상 함수로 만듬으로써, 언제나 동적 바인딩이 제대로 동작하게 만들 수 있습니다.
>
> 실제로 자바의 경우 모든 함수들이 디폴트로 `virtual` 함수로 선언됩니다.
>
> 그렇다면 왜 C++ 에서는 `virtual` 키워드를 이용해 사용자가 직접 `virtual` 로 선언하도록 하였을까요? 그 이유는 가상 함수를 사용하게 되면 약간의 **오버헤드 (overhead)** 가 존재하기 때문입니다.
>
> C++ 컴파일러는 가상 함수가 하나라도 존재하는 클래스에 대해서, **가상 함수 테이블(virtual function table; vtable)**을 만들게 됩니다. 가상 함수 테이블은 전화 번호부라고 생각하시면 됩니다.
>
> 함수의 이름(전화번호부의 가게명) 과 실제로 어떤 함수 (그 가게의 전화번호) 가 대응되는지 테이블로 저장하고 있는 것입니다

<br>

### 순수 가상 함수와 추상 클래스

```cpp
#include <iostream>

class Animal {
 public:
  Animal() {}
  virtual ~Animal() {}
  virtual void speak() = 0;
};

class Dog : public Animal {
 public:
  Dog() : Animal() {}
  void speak() override { std::cout << "왈왈" << std::endl; }
};

int main() {
  Animal* dog = new Dog();

  dog->speak();
}
```

> `Animal` 클래스의 `speak` 함수를 살펴봅시다. 다른 함수들과는 달리, 함수의 몸통이 정의되어 있지 않고 단순히 `= 0;` 으로 처리되어 있는 가상 함수 입니다.
>
> 그렇다면 이 함수는 무엇을 하는 함수 일까요? 그 답은, "무엇을 하는지 정의되어 있지 않는 함수" 입니다. 다시 말해 이 함수는 **반드시 오버라이딩 되어야만 하는 함수** 이지요.
>
> 이렇게, 가상 함수에 `= 0;` 을 붙여서, 반드시 오버라이딩 되도록 만든 함수를 완전한 가상 함수라 해서, **순수 가상 함수(pure virtual function)**라고 부릅니다.
>
> 당연하게도, 순수 가상 함수는 본체가 없기 때문에, 이 함수를 호출하는 것은 불가능합니다. 그렇기 때문에, `Animal` 객체를 생성하는것 또한 불가능입니다.
>
> 따라서 `Animal` 처럼,순수 가상 함수를 최소 한 개 이상 포함하고 있는 클래스는 객체를 생성할 수 없으며, 인스턴스화 시키기 위해서는 이 클래스를 상속 받는 클래스를 만들어서 모든 순수 가상 함수를 오버라이딩 해주어야만 합니다.
>
> 이렇게 순수 가상 함수를 최소 한개 포함하고 있는- 반드시 상속 되어야 하는 클래스를 가리켜 **추상 클래스 (abstract class)**라고 부릅니다.
>
> 추상 클래스 자체로는 인스턴스화 시킬 수 도 없고 (추상 클래스의 객체를 만들 수 없다) 사용하기 위해서는 반드시 다른 누구가 상속 해줘야만 하기 때문이지요. 하지만, 추상 클래스를 '설계도' 라고 생각하면 좋습니다.
>
> 즉, 이 클래스를 상속받아서 사용하는 사람에게 "이 기능은 일반적인 상황에서 만들기 힘드니 너가 직접 특수화 되는 클래스에 맞추어서 만들어서 써라." 라고 말해주는 것이지요

<br>

<br>

## 다중 상속

```cpp
#include <iostream>

class A {
 public:
  int a;

  A() { std::cout << "A 생성자 호출" << std::endl; }
};

class B {
 public:
  int b;

  B() { std::cout << "B 생성자 호출" << std::endl; }
};

class C : public A, public B {
 public:
  int c;
	// 기반 클래스의 생성자 호출 순서
  C() : A(), B() { std::cout << "C 생성자 호출" << std::endl; }
};
int main() { C c; }
```

다중 상속은 `class C : public A, public B {}`와 같이, 단순히 여러 개를 선언해 주는 것으로 할 수 있으며, 생성자에서 호출하는 순서에 따라 그 순서가 결정된다. 즉, ``class C : public B, public A {}`라고 수정해도 같은 결과를 보인다.

<br>

### 다이아몬드 상속, Diamond Inheritance

```cpp
class A {
 public:
  int a;
};

class B {
 public:
  int a;
};

class C : public B, public A {
 public:
  int c;
};
```

 위 코드에서는 기반 클래스 A, B 모두에 같은 이름의 a라는 멤버 변수가 존재할 때 발생하는 문제이다. 다이아몬드 상속은 이와 같은 문제로 부터 발생하는 경우로, 하나의 기반 클래스를 상속받는 파생 클래스 두 개가 있을 때, 만약 또 다른 파생클래스가 먼저의 두 파생클래스들을 상속받는 다면, 이를 다이아몬드 형태로 시각화할 수 있기 때문에 다이아몬드 상속이라고 한다.

```cpp
class Human {
  // ...
};
class HandsomeHuman : public Human {
  // ...
};
class SmartHuman : public Human {
  // ...
};
class Me : public HandsomeHuman, public SmartHuman {
  // ...
};
```

위 다이아몬드 상속을 해결하기 위해서는 `virtual`을 사용하면 된다.

```cpp
class Human {
 public:
  // ...
};
class HandsomeHuman : public virtual Human {
  // ...
};
class SmartHuman : public virtual Human {
  // ...
};
class Me : public HandsomeHuman, public SmartHuman {
  // ...
};
```

> 이러한 형태로 `Human` 을 `virtual` 로 상속 받는다면, `Me` 에서 다중 상속 시에도, 컴파일러가 언제나 `Human` 을 한 번만 포함하도록 지정할 수 있게 됩니다. 참고로, 가상 상속 시에, `Me` 의 생성자에서 `HandsomeHuman` 과 `SmartHuman` 의 생성자를 호출함은 당연하고, `Human` 의 생성자 또한 호출해주어야만 합니다.

<br>

### 다중 상속을 사용하는 경우

예를 들어서 여러분이 **차량(Vehicle)** 에 관련한 클래스를 생성한다고 해봅시다. 차량의 종류로는 땅에서 다니는 차, 물에서 다니는 차, 하늘에서 다니는 차, 우주에서 다니는 차들이 있다고 해봅시다. (차 라고 하기 보다는 운송 수단이 좀 더 적절한 표현이겠네요..)

또한, 이 차량들은 각기 다른 동력원들을 사용하는데, 휘발유를 사용할 수 도 있고, 풍력으로 갈 수 도 있고 원자력으로 갈 수도 있고, 페달을 밟아서 갈 수 도 있습니다.

이러한 차량들을 클래스로 나타내기 위해서, 다중 상속을 활용할 수 있지만 그 전에, 아래와 같은 질문들에 대한 대답을 생각해봅시다.

- `LandVehicle` 을 가리키는 `Vehicle&` 레퍼런스를 필요로 할까? 다시 말해, `Vehicle` 레퍼런스가 실제로는 `LandVehicle` 을 참조하고 있다면, `Vehicle` 의 멤버 함수를 호출하였을 때, LandVehicle 의 멤버 함수가 오버라이드 되서 호출되기를 바라나요?
- `GasPoweredVehicle` 의 경우도 마찬가지 입니다. 만일 `Vehicle` 레퍼런스가 실제로는 `GasPoweredVehicle` 을 참조하고 있을 때, `Vehicle` 레퍼런스의 멤버함수를 호출한다면, `GasPoweredVehicle` 의 멤버 함수가 오버라이드 되서 호출되기를 원하나요?

만일 두 개의 질문에 대한 대답이 모두 **예** 라면 다중 상속을 사용하는 것이 좋을 것입니다. 하지만 그 전에, 몇 가지 고려할 점이 더 있습니다. 만약에 이 차량이 작동하는 환경이 N*N* 개가 있고 (땅, 물, 하늘, 우주 등등), 동력원의 종류가 M*M* 개가 있다고 해봅시다.

이를 위해서, 크게 3 가지 방법으로 이러한 클래스를 디자인 할 수 있습니다. 바로 브리지 패턴 (bridge pattern), 중첩된 일반화 방식 (nested generalization), 다중 상속 입니다. 각각의 방식에는 모두 장단점이 있습니다.

- **브리지 패턴**의 경우 차량을 나타내는 한 가지 카테고리를 아예 멤버 포인터로 만들어버립니다. 예를 들어서 `Vehicle` 클래스의 파생 클래스로 `LandVehicle`, `SpaceVehicle` 클래스들이 있고, `Vehicle` 클래스의 멤버 변수로 어떤 엔진을 사용하는지 가리키는 `Engine*` 멤버 변수가 있습니다. 이 `Engine` 은 `GasPowered`, `NuclearPowered` 와 같은 `Engine` 의 파생 클래스들의 객체들을 가리키게 됩니다. 그리고 런타임 시에 사용자가 `Engine` 을 적절히 설정해주면 됩니다. 이 경우 동력원 이나 환경을 하나 추가하더라도 클래스를 1 개만 더 만들면 됩니다. 즉, 총 $N + M$ 개의 클래스만 생성하면 된다는 뜻입니다.
  하지만 오버라이딩 가지수가 $N + M$ 개 뿐이므로 최대 $N + M$ 개 알고리즘 밖에 사용할 수 없습니다. 만일 여러분이 $N \times M$ 개의 모든 상황에 대한 섬세한 제어가 필요하다면 브리지 패턴을 사용하지 않는 것이 좋습니다. 또한, 컴파일 타임 타입 체크를 적절히 활용할 수 없다는 문제가 있습니다. 예를 들어서 `Engine` 이 페달이고 작동 환경이 우주라면, 애초에 해당 객체를 생성할 수 없어야 하지만 이를 컴파일 타임에서 강제할 방법이 없고 런타임에서나 확인할 수 있게 됩니다. 뿐만 아니라, 우주에서 작동하는 모든 차량을 가리킬 수 있는 기반 클래스를 만들 수 있지만 (`SpaceVehicle` 클래스), 작동 환경에 관계 없이 휘발유를 사용하는 모든 차량을 가리킬 수 있는 기반 클래스를 만들 수 는 없습니다.
- **중첩된 일반화** 방식을 사용하게 된다면, 한 가지 계층을 먼저 골라서 파생 클래스들을 생성합니다. 예를 들어서 `Vehicle` 클래스의 파생 클래스들로 `LandVehicle`, `WaterVehicle`, 등등이 있겠지요. 그 후에, 각각의 클래스들의 대해 다른 계층에 해당하는 파생 클래스들을 더 생성합니다. 예컨대 `LandVehicle` 의 경우 동력원으로 휘발유를 사용한다면 `GasPoweredLandVehicle`, 원자력을 사용한다면 `NuclearPoweredLandVehicle` 클래스를 생성할 수 있겠지요.
  따라서 최대 N \times M*N*×*M* 가지의 파생 클래스들을 생성할 수 있게 됩니다. 따라서 브릿지 패턴에 비해서 좀 더 섬세한 제어를 할 수 있게 됩니다. 왜냐하면 오버라이딩 가지수가 N + M*N*+*M* 이 아닌 N \times M*N*×*M* 이 되기 때문이지요. 하지만 동력원을 하나 더 추가하게 된다면 최대 N*N* 개의 파생 클래스를 더 만들어야 합니다. 뿐만 아니라 앞서 브릿지 패턴에서 나왔던 문제 - 휘발유를 사용하는 모든 차량을 가리킬 수 있는 기반 클래스를 만들 수 없다가 여전히 있습니다. 따라서 만약에 휘발유를 사용하는 차량들에서 공통적으로 사용되는 코드가 있다면 매 번 새로 작성해줘야만 합니다.
- **다중 상속**을 이용하게 된다면, 브리지 패턴 처럼 각 카테고리에 해당하는 파생 클래스들을 만들게 되지만, 그 대신 `Engine*` 멤버 변수를 없애고 동력원과 환경에 해당하는 클래스를 상속받는 파생 클래스들을 최대 $N \times M$ 개 만들게 됩니다. 예를 들어서 휘발유를 사용하며 지상에서 다니는 차량을 나타내는 `GasPoweredLandVehicle` 클래스의 경우 `GasPoweredEngine` 과 `LandVehicle` 두 개의 클래스를 상속받겠지요.
  따라서 이 방식을 통해서 브리지 패턴에서 불가능 하였던 섬세한 제어를 수행할 수 있을 뿐더러, 말도 안되는 조합을 (예컨대 `PedalPoweredSpaceVehicle`) 컴파일 타입에서 확인할 수 있습니다 (애초에 정의 자체를 안하면 되니까요!). 또한 이전에 두 방식에서 발생하였던 **휘발유를 사용하는 모든 차량을 가리킬 수 없다** 문제를 해결할 수 있습니다. 왜냐하면 이제 `GasPoweredEngine` 을 통해서 휘발유를 사용하는 모든 차량을 가리킬 수 있기 때문이지요.

가장 중요한 점은, **위 3 가지 방식 중에서 절대적으로 우월한 방식은 없다**는 것입니다. 상황에 맞게 최선의 방식을 골라서 사용해야 합니다.

다중 상속은 만능 툴이 아닙니다. 실제로 다중 상속을 이용해서 해결해야 될 것 같은 문제도 알고보면 단일 상속을 통해 해결할 수 있는 경우들이 있습니다. 하지만 적절한 상황에 다중 상속을 이용한다면 위력적인 도구가 될 수 있을 것입니다.

