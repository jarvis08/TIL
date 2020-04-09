# Object Oriented Programming

프로그램의 크기가 커짐에 따라 생겨난 패러다임이다. 기존의 절차(Procedure) 지향 언어는 중요 부분을 하나의 procedure로 만듦으로서, 전체 코드를 쪼갰다.

### 객체 지향이란?

먼저 아래의 예제를 통해 객체 지향을 이해해 보겠다.

```cpp
#include <iostream>

typedef struct Animal {
//	char name[30] = "lion";
	char name[30];
    int age;
    int health;
} Animal;

void create_animal(Animal *animal) {
	std::cin >> animal->name;
	animal->age = 5;
    animal->health = 100;
}

void play(Animal *animal) {
  animal->health += 10;
  std::cout << animal->health << std::endl;
  std::cout << animal->name << std::endl;
}

int main() {
    Animal *list[10];
    list[0] = new Animal;
    create_animal(list[0]);
    play(list[0]);
    delete list[0];
}
```

위 코드 블록에서는 `play` 함수가 `animal` 객체를 사용하도록 했다. 하지만, 현실 세계를 보자면 `animal`이 `play`를 하는 것이 맞다. 따라서 `Animal` 구조체 안에 `play` method를 만들어 준다면, `animal.play()` 형태로 `animal`이 `play`할 수 있도록 할 수 있다.

>  객체가 현실 세계에서의 존재하는 것들을 나타내기 위해서는 **추상화(abstraction)** 라는 과정이 필요합니다. 추상화는 컴퓨터 상에서 현실 세계를 100% 나타낼 수 없는 것이기 때문에, 적절하게 컴퓨터에서 처리할 수 있도록 바꾸는 것이다. 예를 들어서, 핸드폰의 경우 '전화를 한다', '문자를 보낸다' 와 같은 것들은 `핸드폰이 하는 것` 이므로 함수로 추상화시킬 수 있다. 또한, `핸드폰의 상태`를 나타내는 것들, 예를 들어서 자기 자신의 전화 번호나 배터리 잔량 같은 것은변수로 추상화시킬 수 있습니다. 참고로, 이러한 객체의 변수나 함수들을 보통 **인스턴스 변수(instance variable)** 와 **인스턴스 메소드(instance method)** 라고 부르게 된다.

```cpp
animal.age += 100;
animal.health += 10;
animal.increase_health(100);
```

외부에서 직접 인스턴스 변수의 값을 바꿀 수 없으며, `animal.increase_health(100)`와 같이, 항상 인스턴스 메소드를 통해서 간접적으로 조절하는 것을 **캡슐화(Encapsulation)** 라고 한다.

<br>

<br>

## Class

C++에서는 객체 지향 프로그래밍을 위해 Class를 사용한다.

```cpp
class Animal {
 private:
  int food;
  int weight;

 public:
  void set_animal(int _food, int _weight) {
    food = _food;
    weight = _weight;
  }
  void increase_food(int inc) {
    food += inc;
    weight += (inc / 3);
  }
  void view_stat() {
    std::cout << "이 동물의 food   : " << food << std::endl;
    std::cout << "이 동물의 weight : " << weight << std::endl;
  }
};
```

> 위는 `Animal` 이라는 클래스를 나타낸 것으로 `Animal` 클래스를 통해서 생성될 임의의 객체에 대한 설계도라고 볼 수 있습니다. 즉, `Animal` 클래스를 통해서 생성될 객체는 `food, weight` 라는 변수가 있고, `set_animal, increase_food, view_stat` 이라는 함수들이 있는데, `Animal` 클래스 상에서 이들을 지칭할 때 각각 **멤버 변수(member variable) 과 멤버 함수(member function)** 라고 부릅니다.
>
> 즉, 인스턴스로 생성된 객체에서는 인스턴스 변수, 인스턴스 함수, 그리고 그냥 클래스 상에서는 멤버 변수, 멤버 함수 라고 부르는 것입니다. 멤버 변수와 멤버 함수는 실재 하는 것이 아니지요. 인스턴스가 만들어져야 비로소 세상에 나타나는 것입니다. 즉, 설계도 상에 있다고 해서 아파트가 실제로 존재하는 것이 아닌 것 처럼 말이지요.
>
> 위는 각 멤버 변수들의 값을 설정하는 부분인데요, 여기서 `food` 와 `weight` 는 누구의 것일까요? 당연하게도, 객체 자신의 것입니다. 그렇기 때문에 `food` 와 `weight` 가 누구 것인지 명시할 필요 없이 그냥 `food, weight` 라고 사용하면 됩니다.

<br>

### Private과 Puplic

만약 `private`로 설정된 멤버 함수 혹은 변수들을 외부에서 사용하려하면 접근이 불가능 하다며 에러가 발생한다. 만약 특별히 명시를 하지 않는다면 `private`로 지정되니 주의한다. `private`로 설정된 멤버 함수 및 변수들은 `public`으로 설정된 멤버 함수들을 통해서만 접근 및 사용이 가능하다.

```cpp
#include <iostream>

class Animal {
private:
    int age = 29;
    void bark() {
        std::cout << "work! work!" << std::endl;
    }
public:
    void call() {
        bark();
        std::cout << age << std::endl;
    }
};
int main() {
    auto *a = new Animal;
    a->call();
    delete a;
}
```

```
work! work!
29
```

<br>

### this

 `this` 라는 것이 C++ 언어 차원에서 정의되어 있는 키워드 인데, 이는 객체 자신을 가리키는 포인터의 역할을 한다. 즉, 이 멤버 함수를 호출하는 객체 자신을 가리킨다는 것이다.

```cpp
Marine& Marine::be_attacked(int damage_earn) {
  hp -= damage_earn;
  if (hp <= 0) is_dead = true;

  return *this;
}
```

위와 같이 작성된 코드는, 아래와 동일한 역할을 한다.

```cpp
Marine& Marine::be_attacked(int damage_earn) {
  this->hp -= damage_earn;
  if (this->hp <= 0) this->is_dead = true;

  return *this;
}
```

실제로 **모든 멤버 함수 내에서는 `this` 키워드가 정의되어 있으며** 클래스 안에서 정의된 함수 중에서 `this` 키워드가 없는 함수는 `static` 함수 뿐이다.

 `be_attacked()` 함수는 `*this`를 통해 자기 자신 객체를 리턴하므로, 아래와 같이 작성한다면 함수를 한번에 여러번 사용하도록 선언할 수 있다. `marine1.attack()`은 int 타입을 반환한다.

```cpp
marine2.be_attacked(marine1.attack()).be_attacked(marine1.attack());
```

<br>

### Reference 받아서 외부에서 멤버 변수 수정하기

```cpp
#include <iostream>

class A {
  int x;

 public:
  A(int c) : x(c) {}

  int& access_x() { return x; }
  int get_x() { return x; }
  void show_x() { std::cout << x << std::endl; }
};

int main() {
  A a(5);
  a.show_x();

  // 유일하게 제대로 동작한 코드
  int& c = a.access_x(); // int& c = x 와도 동일한 역할
  c = 4;
  a.show_x();

  int d = a.access_x(); // 단순하게 값의 복사가 발생하며, x와 분리됨
  d = 3;
  a.show_x();

  int& e = a.get_x(); // 에러 발생, 아래 결과에서는 제외하고 컴파일
  e = 2;
  a.show_x();

  int f = a.get_x();
  f = 1;
  a.show_x();
}
```

```
5
4
4
4
```

위의 `get_x`를 통해서는 '값' 자체인 literal을 받아올 뿐, 주소가 존재하지 않으므로 에러가 발생한다. 레퍼런스가 아닌 타입을 리턴하는 경우는 '값' 의 복사가 이루어지기 때문에 임시 객체가 생성되는데, 임시객체의 레퍼런스를 가질 수 없기 때문이다.

<br>

<br>

## 상수 함수

 **변수들의 값을 바꾸지 않고 읽기만** 하는, 마치 상수 같은 멤버 함수인 **`const` 함수**를 선언할 수 있다.

```cpp
// 상수 멤버 함수
#include <iostream>

class Marine {
  static int total_marine_num;
  const static int i = 0;
  ...
  int attack() const; // const 함수 선언
};

// const 함수 선언, 기존에 const를 붙여준 것
int Marine::attack() const { return default_damage; }
```

상수 함수는 다른 멤버 함수를 호출할 때에도, 오직 다른 상수 함수를 호출하는 것만 가능하다.

<br>

### mutable 멤버 변수

`const` 함수는 변수들의 값을 변경할 수 없지만, `mutable`로 선언된 내용만큼은 변경할 수 있다.

```cpp
#include <iostream>

class A {
  mutable int mdata_;
  int data_;

 public:
  A(int data) : data_(data) {}
  void DoSomething(int x) const {
    mdata_ = x;  // 가능
    data_ = x;  // 불가능
  }
};
```

이는 특별한 경우들에만 사용이 된다. 예를들어, 데이터베이스에서 자주 사용하는 사용자 정보를 가져오는 cache 개념이 있다고 하자. 데이터베이스를 읽는 `get_user_info()` 함수의 경우 DB를 훼손하면 안되므로 `const`로 선언되어 있다. 그런데 `const` 함수는 값을 변경할 수 없으며, `const`로 선언된 `get_user_info()` 함수는 cache 내용을 업데이트를 시켜야한다. 이런 경우, cache 내용을 저장하는 변수를 `mutable`로 선언하면 `const` 함수인 `get_user_info()`에서도 cache 변수를 변경할 수 있다.
