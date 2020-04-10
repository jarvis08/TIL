# Inheritance

```cpp
// 기반 클래스
class Base {
  std::string s;

 public:
  Base() : s("기반") {  std::cout << "기반 클래스" <<  std::endl; }

  void what() {  std::cout << s <<  std::endl; }
};

// 파생 클래스
class Derived : public Base {
  std::string s;

 public:
  Derived() : Base(), s("파생") {
     std::cout << "파생 클래스" <<  std::endl;

    // Base 에서 what() 을 물려 받았으므로
    // Derived 에서 당연히 호출 가능하다
    what();
  }
};
```

 상속을 받을 때에는, `class Derived : public Base {}`와 같이 부모(기반) 클래스를 colon(`:`) 이후에 명시에 해주어야 한다.

 또한, 초기화 시 파생 클래스의 constructor를 정의할 때, `Derived() : Base(), s("파생") {}`와 같이 기반 **클래스의 생성자를 가장 먼저 호출해야 한다**. 만약 임의로 customized constructor를 호출해주지 않는다면, default constructor가 호출된다.

<br>

### Overriding

 같은 이름의 멤버 함수 혹은 멤버 변수가 기반/파생 클래스 모두에 작성되어 있다면, 오버라이딩이 발생한다. 컴파일 시 기반/파생 관계에 있지만, 컴파일러는 두 클래스에 정의된 함수 및 변수를 다른 것으로 취급한다. 즉, 파생 클래스의 인스턴스에서 어떤 멤버 함수를 호출했을 때 만약 파생클래스에 해당 함수가 존재한다면 그 함수를 호출하게 되고, 없다면 기반 클래스에서 탐색 후 호출한다.

<br>

### Protected

 그런데 아무리 상속을 받았다 할 지라도, 파생 클래스가 기반 클래스의 `private` 영역을 직접 조작할 수는 없다. C++에서는 이러한 경우를 위해 `private`과 `public`중간 위치에 있는 접근 지시자 `protected`를 제공한다. 이 키워드는, **상속받는 클래스에서는 접근이 가능한 `private`**라고 볼 수 있다.

<br>

### 상속 형태

 위 코드에서 상속 받음을 선언하기 위해 `class Derived : public Base {}`와 같이 작성했는데, 그 중에서도 `public`은 기반 클래스의 데이터를 어떻게 불러올 것인지를 설정한 것이다.

- `public`: 기반 클래스의 접근 지시자들이 선언된 대로, 변동 없이 작동한다. 즉, 파생 클래스 입장에서 `public` 은 그대로 `public` 이고, `protected` 는 그대로 `protected` 이고, `private` 은 그대로 `private`으로써 작동한다.
- `protected`: 모든 것이 동일하지만, `public`으로 선언된 내용들이 `protected`로 변경된다.
- `private`: 파생 클래스에게 기반 클래스의 모든 접근 지시자들이 `private`로 변경된다.

<br>

### Exercise

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

  // 복사 생성자
  Employee(const Employee& employee) {
    name = employee.name;
    age = employee.age;
    position = employee.position;
    rank = employee.rank;
  }

  // 디폴트 생성자
  Employee() {}

  void print_info() {
    std::cout << name << " (" << position << " , " << age << ") ==> "
              << calculate_pay() << "만원" << std::endl;
  }
  int calculate_pay() { return 200 + rank * 50; }
};

// Inherit
class Manager : public Employee {
  int year_of_service;

 public:
  Manager(std::string name, int age, std::string position, int rank, int year_of_service)
      : Employee(name, age, position, rank), year_of_service(year_of_service) {}

  // 복사 생성자
  Manager(const Manager& manager)
      : Employee(manager.name, manager.age, manager.position, manager.rank) {
    year_of_service = manager.year_of_service;
  }

  // 디폴트 생성자
  Manager() : Employee() {}

  int calculate_pay() { return 200 + rank * 50 + 5 * year_of_service; }
  void print_info() {
    std::cout << name << " (" << position << " , " << age << ", "
              << year_of_service << "년차) ==> " << calculate_pay() << "만원"
              << std::endl;
  }
};

class EmployeeList {
  int alloc_employee;  // 할당한 총 직원 수

  int current_employee;  // 현재 직원 수
  int current_manager;   // 현재 매니저 수

  Employee** employee_list;  // 직원 데이터
  Manager** manager_list;    // 매니저 데이터

 public:
  EmployeeList(int alloc_employee) : alloc_employee(alloc_employee) {
    employee_list = new Employee*[alloc_employee];
    manager_list = new Manager*[alloc_employee];

    current_employee = 0;
    current_manager = 0;
  }
  void add_employee(Employee* employee) {
    employee_list[current_employee] = employee;
    current_employee++;
  }
  void add_manager(Manager* manager) {
    manager_list[current_manager] = manager;
    current_manager++;
  }
  int current_employee_num() { return current_employee + current_manager; }

  void print_employee_info() {
    int total_pay = 0;
    for (int i = 0; i < current_employee; i++) {
      employee_list[i]->print_info();
      total_pay += employee_list[i]->calculate_pay();
    }
    for (int i = 0; i < current_manager; i++) {
      manager_list[i]->print_info();
      total_pay += manager_list[i]->calculate_pay();
    }
    std::cout << "총 비용 : " << total_pay << "만원 " << std::endl;
  }
  ~EmployeeList() {
    for (int i = 0; i < current_employee; i++) {
      delete employee_list[i];
    }
    for (int i = 0; i < current_manager; i++) {
      delete manager_list[i];
    }
    delete[] employee_list;
    delete[] manager_list;
  }
};
int main() {
  EmployeeList emp_list(10);
  emp_list.add_employee(new Employee("노홍철", 34, "평사원", 1));
  emp_list.add_employee(new Employee("하하", 34, "평사원", 1));
  emp_list.add_manager(new Manager("유재석", 41, "부장", 7, 12));
  emp_list.add_manager(new Manager("정준하", 43, "과장", 4, 15));
  emp_list.add_manager(new Manager("박명수", 43, "차장", 5, 13));
  emp_list.add_employee(new Employee("정형돈", 36, "대리", 2));
  emp_list.add_employee(new Employee("길", 36, "인턴", -2));
  emp_list.print_employee_info();
  return 0;
}
```

<br>

<br>

## 상속의 용도

상속은 단순히 Ctrl+C, Ctrl+V를 하기 위함이 아니다. 이는 표면적인 유용함을 가져다줄 뿐, 정말 중요한 것은 객체의 추상화를 매우 효율적으로 할 수 있다는 것이다. 위 Exercise에서 `Employee`를 상속받은 `Manager`는 `Employee`의 모든 기능을 수행할 수 있으므로, `Manager` 클래스의 인스턴스들은 모두 `Employee`라고 볼 수 있다. 이를 `is-a` 관계라고 말한다.

<br>

### is-a와 has-a

> 실제 세상에서 `is a` 관계로 이루어진 것들은 수 없이 많습니다. 예를 들어, '사람' 이라는 클래스가 있다면, '프로그래머는 사람이다 (A programmer is a human)' 이므로, 만일 우리가 프로그래머 클래스를 만든다면 사람 이라는 클래스를 상속 받을 수 있도록 구성할 수 있습니다.
>
> 이를 통해서 상속의 또 하나의 중요한 특징을 알 수 있습니다. 바로 클래스가 파생되면 파생될 수 록 좀 더 **특수화 (구체화;specialize)** 된다는 의미 입니다. 즉, `Employee` 클래스가 일반적인 사원을 위한 클래스 였다면 `Manager` 클래스 들은 그 일반적인 사원들 중에서도 좀 더 특수한 부류의 사원들을 의미하게 됩니다.
>
> 또, `BankAccount` 도 일반적인 은행 계좌를 위한 클래스 였다면, 이를 상속 받는 `CheckingAccount, SavingsAccount` 들은 좀 더 구체적인 클래스가 되지요. 반대로, 기반 클래스로 거슬러 올라가면 올라갈 수 록 좀 더 **일반화 (generalize)** 된다고 말합니다.
>
> 그렇다면 모든 클래스들의 관계를 `is - a` 로만 표현할 수 있을까요? 당연히 그렇지 않습니다. 어떤 클래스들 사이에서는 `is - a` 대신에 `has - a` 관계가 성립하기도 합니다. 예를 들어서, 간단히 자동차 클래스를 생각해봅시다. 자동차 클래스를 구성하기 위해서는 엔진 클래스, 브레이크 클래스, 오디오 클래스 등 수 많은 클래스들이 필요합니다. 그렇다고 이들 사이에 `is a` 관계를 도입 할 수 없습니다. (자동차 `is a` 엔진? 자동차 `is a` 브레이크?) 그 대신, 이들 사이는 `has - a` 관계로 쉽게 표현할 수 있습니다.

<br>

### Up Casting & Down Casting

```cpp
// Up Casting
#include <iostream>
#include <string>

class Base {
  std::string s;

 public:
  Base() : s("기반") { std::cout << "기반 클래스" << std::endl; }

  void what() { std::cout << s << std::endl; }
};
class Derived : public Base {
  std::string s;

 public:
  Derived() : s("파생"), Base() { std::cout << "파생 클래스" << std::endl; }

  void what() { std::cout << s << std::endl; }
};
int main() {
  Base p;
  Derived c;

  std::cout << "=== 포인터 버전 ===" << std::endl;
  // Base 객체를 가리키도록하는 p_c 포인터가 Derived 객체인 c를 가리킴
  Base* p_c = &c;
  p_c->what();

  return 0;
}
```

`Derived is a Base`가 성립하는 코드이며, `Derived` 객체 `c` 도 어떻게 보면 `Base` 객체이기 때문에 `Base` 객체를 가리키는 포인터가 `c` 를 가리켜도 무방하다. 즉, 파생 클래스에서 기반 클래스로 캐스팅하는 것이 가능하며, 이를 Up Casting이라고 한다. 그런데, `p`는 `Base`를 가리키는 포인터이므로, `Derived` 객체 `c` 내에 존재하는 `Base`에게 상속받은 영역만을 사용할 수 있다.

하지만 아래와 같은 Down Casting에서는 에러가 발생한다. 아래 코드는 `Derived` 포인터가 `Base` 객체를 가리키도록 하고 있다.

```cpp
 C/C++ 확대 축소
#include <iostream>
#include <string>

class Base {
  std::string s;

 public:
  Base() : s("기반") { std::cout << "기반 클래스" << std::endl; }

  void what() { std::cout << s << std::endl; }
};
class Derived : public Base {
  std::string s;

 public:
  Derived() : s("파생"), Base() { std::cout << "파생 클래스" << std::endl; }

  void what() { std::cout << s << std::endl; }
};
int main() {
  Base p;
  Derived c;

  std::cout << "=== 포인터 버전 ===" << std::endl;
  // Derived 객체를 가리키도록 하는 포인터가 Base 객체를 가리킴
  Derived* p_p = &p;
  p_p->what();

  return 0;
}
```

 `Derived` 객체를 가리키는 포인터는 객체가 `Derived`라고 생각하지만, `Derived`의 `what()` 함수가 정의되지 않았으므로 사용할 수 없다. 이는 `Derived* p_c = static_cast<Derived*>(p_p);`와 같이 강제로 타입 변환을 한다 해도, 여전히 에러가 발생하므로 다운 캐스팅은 특별한 경우가 아니라면 권장하지 않는다.