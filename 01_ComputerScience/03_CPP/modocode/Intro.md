# 씹어먹는 C++

출처: https://modoocode.com/135

`#include <iostream>`: standard libarary의 header file을 link

<br>

### static

1. 지역 변수와 같이, 선언된 함수 내에서만 사용이 가능
2. C의 경우 함수의 맨 처음에 선언해야 하며, C++은 어디서든 가능
3. 전역 변수와 같이, 프로그램이 종료될 때 가지 메모리 공간에 존재
4. 프로그램을 실행시킬 때 초기화되며, 값을 지정하지 않을 경우 `0`으로 초기화

```cpp
#include <iostream>

void simpleFunc();

int main() {
    int i;
    for(i=0; i<3; i++)
        simpleFunc();
    return 0;
}

void simpleFunc() {
    static int num1 = 0;
    int num2 = 0;
    num1++, num2++;
    std::cout << "static: " << num1 << " / local: " << num2 << std::endl;
}
```

```
static: 1 / local: 1
static: 2 / local: 1
static: 3 / local: 1
```

<br>

### Literal

프로그래밍 언어에서 **리터럴(literal)**이란, 소스 코드 상에서 고정된 값을 가지는 것이다. 특히, C 언어의 경우 큰 따옴표(`"`) 로 묶인 것들을 **문자열 리터럴(string literal)**이라 부른다. 리터럴이 보관되는 곳은 **오직 읽기만 가능한 곳**이며, 만일 변경하려고 하는 시도가 있다면 프로그램이 강제로 종료된다.

```cpp
char *pstr = "goodbye";
pstr[1] = 'a';
```

위와 같이 `"goodbye"`라는 리터럴을 포인터로 가리킨 후, `pstr[1] = 'a'`을 통해 수정하고자 한다면 에러가 발생한다.

```cpp
char str[] = "hello";
```

이는 그냥 `str` 이라는 배열에 hello 라는 문자열을 `{'h', 'e', 'l', 'l', 'o', '\0'}` 형태로 복사한다. 따라서 위 배열은 text segment가 아니라, 아니라 스택(stack)이라는 메모리 수정이 가능한 영역에 정의되며, 위의 `str`  배열 안의 문자들은 수정이 가능하다.

<br>

<br>

## Overloading

C++에서는 동일한 함수 이름을 여러개 사용해도 괜찮다. 단, **인자가 다른 함수일 경우에만** 괜찮다.

```cpp
#include <iostream>

void print(int x) { std::cout << "int : " << x << std::endl; }
void print(char x) { std::cout << "char : " << x << std::endl; }
void print(double x) { std::cout << "double : " << x << std::endl; }

int main() {
  int a = 1;
  char b = 'c';
  double c = 3.2f;

  print(a);
  print(b);
  print(c);

  return 0;
}
```

<br>

### Compiler에서의 overloading 처리 과정

C++ compiler에서의 함수 오버로딩 과정은 다음과 같다. 만약에 컴파일러가 아래 과정을 통과하던 중, 일치하는 함수를 찾을 수 없거나, 같은 단계에서 두 개 이상이 일치하는 경우에 **모호하다 (ambiguous)** 라고 판단하고 오류를 발생시킨다.

1. 자신과 타입이 정확히 일치하는 함수를 찾는다.
2. 정확히 일치하는 타입이 없는 경우, 아래와 같은 형변환을 통해서 일치하는 함수를 찾아본다.
    - `Char, unsigned char, short` 는 `int` 로 변환된다.
    - `Unsigned short` 는 `int` 의 크기에 따라 `int` 혹은 `unsigned int` 로 변환된다.
    - `Float` 은 `double` 로 변환된다.
    - `Enum` 은 `int` 로 변환된다.
3. 위와 같이 변환해도 일치하는 것이 없다면, 아래의 좀더 포괄적인 형변환을 통해 일치하는 함수를 찾는다.
    - 임의의 숫자(numeric) 타입은 다른 숫자 타입으로 변환된다. (예를 들어 `float -> int)``
    - ``Enum` 도 임의의 숫자 타입으로 변환된다 (예를 들어 `Enum -> double)`
    - 0` 은 포인터 타입이나 숫자 타입으로 변환된 0 은 포인터 타입이나 숫자 타입으로 변환된다` 
    -  `포인터는 `void` 포인터로 변환된다.
4. 유저 정의된 타입 변환으로 일치하는 것을 찾는다.

<br>

<br>

<br>

# Namespace

1. header file에 namespace를 명시

    ```cpp
    // header1.h
    namespace header1 {
    int foo();
    void bar();
    }
    
    // header2.h
    namespace header2 {
    int foo();
    void bar();
    }
    ```

2. `.cpp` 파일에서 사용

    ```c++
    // main.cpp
    #include "header1.h"
    
    namespace header1 {
    int func() {
      foo();           // 알아서 header1::foo() 실행
      header2::foo();  // header2::foo() 가 실행
    }
    }
    ```

3. (필요시) `using namespace (이름);`를 사용하여 간편화

    ```cpp
    using namespace std;
    
    int main() {
        cout << "간편하다." << endl;
        return 0;
    }
    ```

    하지만, standard libarary(`std`)의 경우 다양한 함수들이 존재하며, C++ 버전에 따라 새로 추가되는 내용도 많다. 따라서 과거에 작동했지만, 다음 버전에서는 충돌이 나는 경우가 발생할 수 있으므로 권장하지 않는다.

<br>

### namespace 이름 없이 사용하기

```cpp
#include <iostream>

namespace {
// 이 함수는 이 파일 안에서만 사용할 수 있습니다.
// 이는 마치 static int OnlyInThisFile() 과 동일합니다.
int OnlyInThisFile() {}

// 이 변수 역시 static int x 와 동일합니다.
int only_in_this_file = 0;
}
```

 헤더파일을 통해서 위 파일을 받았다 하더라도, 저 익명의 `namespace` 안에 정의된 모든 것들은 사용할 수 없다.

<br>

<br>

<br>

# Reference

레퍼런스는 어떤 변수를 참고하는, 다른 이름을 부여하는 것과 같다.

```c++
int main() {
    int number = 1;
    int& ref = number;
    ref = 3;
}
```

을 작성할 경우, `number`에는 3의 값이 저장됩니다. 주의할 것은 reference는 변경될 수 없다는 점이다. 아래와 같이 실행할 경우, `b`의 값이 `a`에 저장되는 것이며, `a`의 주소와 `b`의 주소가 같아지는 것이 아니다.

```c++
int a = 10;
int &ref = a;
int b = 3;
ref = b;
```

Reference와 pointer의 차이는 아래를 통해 확인할 수 있다.

```c++
#include <iostream>

int main() {
    int number = 10;
    int& ref = number;
    int* p = &number;
    int* p2 = &number;
    for (int i=0; i<5; i++){
        ref++;
        (*p)++;
        *p = *p + 1;
        p2++;
        std::cout << number << std::endl;
        std::cout << p << std::endl;
        std::cout << p2 << std::endl;
    }
}
```

```
13
0x7ffc7f602458
0x7ffc7f60245c
16
0x7ffc7f602458
0x7ffc7f602460
19
0x7ffc7f602458
0x7ffc7f602464
22
0x7ffc7f602458
0x7ffc7f602468
25
0x7ffc7f602458
0x7ffc7f60246c
```

실행 결과, `ref++`에 의해 `number`는 1이 추가될 것이며, `p2++`의 경우 `p2`가 이상한 주소를 가리키게 한다.

<br>

### Reference의 reference

`number`를 참조하는 `&ref1`과, `&ref1`을 참조하는 `ref2`가 있을 때 포인터와의 차이를 알 수 있다.

```cpp
#include <iostream>

int main() {
    int number = 3;
    // reference 이용
    int& ref1 = number;
    int& ref2 = ref1;
    
    // pointer 이용
    int* ptr1 = &number;
    int** ptr2 = &ptr1;
}
```

위와 같이 사용해야 모든 reference와 pointer들이 `number` 변수를 가리키도록 할 수 있다.

<br>

### 함수 내에서, 함수 외의 변수 수정

```c++
#include <iostream>

int by_ref(int &p) {
  p = 3;
  return 0;
}

int by_ptr(int *p) {
  *p = 3;
  return 0;
}

int main() {
  int ref_num = 5;
  int ptr_num = 5;

  std::cout << ref_num << std::endl;
  std::cout << ptr_num << std::endl;
  by_ref(ref_num);
  by_ptr(&ptr_num);
  std::cout << ref_num << std::endl;
  std::cout << ptr_num << std::endl;
}
```

```
5
5
3
3
```

<br>

### `cin`과 `scanf`

C에서는 어떤 변수의 값을 다른 함수에서 변경시키기 위해 포인터를 사용했으며, 이는 `scanf` 함수에서도 알 수 있다.

```c
scanf("%d", &user_input);
```

C++에서 사용하는 `cin` 또한 주소값을 필요로하는 것은 동일하다. 하지만, `cin`의 경우 파라미터를 reference로 받도록 되어 있기 때문에, 다음과 같이 작성하는 것 만으로도 충분하다.

```cpp
std::cin >> user_input;
```

<br>

### Literal 참조

만약 아래와 같이 참조를 시도한다면, 리터럴이 수정될 여지가 있기 때문에 에러가 발생한다.

```cpp
int &ref = 4;
```

하지만, `const`를 사용한다면 리터럴도 참조할 수 있다.

```cpp
#include <iostream>

int main() {
    const int& ref = 4;
    int a = ref;
    std::cout << a << std::endl;
}
```

```
4
```

위 코드는 `int a = 4`를 사용하는 것과 동일한 역할을 하게 된다.

<br>

### 배열 참조

아래 코드 블록과 같이 레퍼런스의 배열을 생성하려고 하면 에러가 발생한다.

```cpp
int a, b;
int& arr[2] = {a, b};
```

왜냐하면, 레퍼런스는 주소값을 갖지 않기 때문이다. 위 코드는 `a`와 `b`를 묶어 배열로 만들고, 이를 `arr`라는 배열에 저장하려고 한다. 하지만 레퍼런스는 주소값을 가질 수 없으므로, 새로운 구조체를 저장할 수 없다.

 따라서 `a`와 `b`를 원소로 갖는 배열은 앞에서 먼저 선언한 후, 해당 배열을 참조하는 레퍼런스를 생성해야 한다. 배열을 참조하기 위해서는, 참조하는 **배열의 크기를 명시**해주어야 한다.

```cpp
#include <iostream>

int main() {
  int arr[3] = {1, 2, 3};
  int(&ref)[3] = arr;

  ref[0] = 2;
  ref[1] = 3;
  ref[2] = 1;

  std::cout << arr[0] << arr[1] << arr[2] << std::endl;
  return 0;
}
```

```
231
```

만약 배열이 2차원 형태를 가질 경우에도 아래와 같이 명시해 주어야 한다.

```cpp
int arr[3][2] = {1, 2, 3, 4, 5, 6};
int (&ref)[3][2] = arr;
```

따라서 배열을 가리켜야 할 때에는 `int *p` 하나로 모든 1차원 배열들을 가리킬 수 있는 포인터를 사용할 것을 권장한다.

<br>

### Reference 반환

아래의 코드 블록에서, 만약 `fn1(x)`를 사용할 경우 `x`의 값이 반환될 뿐 `x` 자체가 반환되지 않는다. 즉, literal 값인 1이 반환되므로, `++`을 사용하려 하면 에러가 발생한다. 하지만 `fn2(x)`를 사용할 경우, `x`의 주소값을 반환하므로 `x` 변수에 `++`이 적용된다.

```cpp
#include <iostream>

int fn1(int &a) { return a; }
int &fn2(int &a) { return a; }

int main() {
	int x = 1;
	std::cout << fn2(x)++ << std::endl;
	std::cout << x << std::endl;
}
```

```
1
2
```

<br>

### Reference가 메모리에 존재하는가?

Reference는 함수 호출로 인해 호출 스택이 달라질 때, 해당 메모리에 접근하기 위해 주소가 필요하다. 따라서 이러한 경우 주소 메모리 공간이 필요하다. 만약 호출 스택이 같다면, 바로 접근할 수 있으며, 별도의 주소 메모리 공간이 필요 없다.

1. 메모리에 존재하는 경우란, 즉 참조가 유효한 경우, 같은 함수 내에 있는 경우 혹은 runtime이 해당 함수를 실행하는 때 일 경우라는 뜻이죠. 간단한 예로 main 에 선언된 참조는 선언 순간부터 main함수가 끝날 때 까지 대상 변수와 같은 저장 공간에 공존해 있습니다.
2. 여기서! 메모리에 존재하지 않는 경우란, 즉 해당 함수가 끝나고 runtime 이 다음 영역으로 갔을때를 말합니다. 위의 예제대로 `int& a`는 `fn1`이건 `fn2` 이건, 호출시 `x`의 저장공간에 세내고 묵고있다가 함수값 반환과 동시에 방빼고 쫓겨납니다. 여기서 차이는 반환값이죠. `fn1`의 반환값은 참조가 아니기 때문에 해당 소스에서는 함수값 반환과 동시에 `x`는 방혼자쓰는 쏠로가 됩니다. 하지만 `fn2`의 경우 반환값이 참조형식이기 때문에 `a`는 사라졌을지언정, `fn2`가 `x`의 참조로서 남아있습니다. 정확히는 `fn2(x)`인데, 이 동거인은 `main`함수에 반환되었기 때문에 `main`이 끝날때까지 살아있습니다. 간단한 실험으로 한번 `fn2(x)`를 `for`문에 `++`로 다섯번정도 증감시켜 보세요. 그 후, `x`나 `fn2(x)`를 출력하면 값에 5가 증가한 것을 보실 수 있습니다. 거기에 또! `fn2`에 `y`라는 새로운 변수를 넣어서 `fn2(y)`라는 참조연산자를 새로만들고, `fn2(x)`랑 섞어서 이것저것 증감해보세요. 컴파일해 보시면 아시겠지만, `y`는 `y`대로, `x`는 `x`대로 값이 증감되있는 것을 확인하실 수 있으실 겁니다.

<br>

<br>

<br>

# Memory

## Heap

> 메모리를 관리하는 문제는 언제나 중요한 문제입니다. 프로그램이 정확하게 실행되기 위해서는 컴파일 시에 모든 변수의 주소값이 확정되어야만 했습니다. 하지만, 이를 위해서는 프로그램에 많은 제약이 따르기 때문에 프로그램 실행 시에 자유롭게 할당하고 해제할 수 있는 **힙(heap)** 이라는 공간이 따로 생겼습니다.

Heap 공간은  C에서는 `malloc`과 `free`를 사용하여 조작했으며, C++에서는 `new`와 `delete`를 사용한다.

```cpp
#include <iostream>

int main() {
    int* p = new int;
    *p = 10;
    
    std::cout << *p << std::endl;
    std::cout << *p << std::endl;
    delete ptr;
    delete p;
    return 0;
}
```

<br>

### 배열 할당

배열은 아래와 같이 사용할 수 있다.

```cpp
#include <iostream>

int main() {
  int arr_size = 5;
  int *list = new int[arr_size];
    
  for (int i = 0; i < arr_size; i++) {
    list[i] = i;
  }
  for (int i = 0; i < arr_size; i++) {
    std::cout << i << "th element of list : " << list[i] << std::endl;
  }
  delete[] list;
  return 0;
}
```

```
0th element of list : 0
1th element of list : 1
2th element of list : 2
3th element of list : 3
4th element of list : 4
```

`new`로 선언했다면 `delete`를 사용하고, `new[]`를 선언했다면 `delete[]`를 사용하여 할당을 해제한다.

<br>

### 예시

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

<br>

<br>

## Scope

C++에서는 C와는 다르게, 변수 할당을 맨 위에서 하지 않아도 된다. 또한, scope를 활용하면 아래와 같이 같은 이름의 변수를 구분하여 사용할 수 있다.

```cpp
#include <iostream>

int main() {
    int a = 4;
    {
      std::cout << "외부 변수 a = " << a << std::endl;
      int a = 3;
      std::cout << "내부 변수 a = " << a << std::endl;
    }

    std::cout << "외부 변수 a = " << a << std::endl;
  return 0;
}
```

```
외부 변수 a = 4
내부 변수 a = 3
외부 변수 a = 4
```

<br>

### `for`문의 Counter

이는 `for`문에서도 알 수 있다. 아래 코드 블록 내부 `for`문의 `(int i =0; i < 5; i++)`는 바깥의 `i`와 다른 변수로 정의된다.

```cpp
#include <iostream>

int main() {
    int i = 10;
    std::cout << i << std::endl;
    for(int i=0; i < 3; i++){
        std::cout << i << std::endl;
    }
    std::cout << i << std::endl;
  return 0;
}
```

```
10
0
1
2
10
```

<br>

<br>

<br>

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

<br>

## Constructor

### Default Constructor

아래는 코드블록에서는 인스턴스 생성 시 자동으로 값을 할당하도록 default constructor를 사용했다.

```cpp
#include <iostream>

class Date {
  int year_;
  int month_;
  int day_;

 public:
    Date();
    void ShowDate();

};

Date::Date() {
    year_ = 2019;
    month_ = 4;
    day_ = 7;
}

void Date::ShowDate() {
  std::cout << "오늘은 " << year_ << " 년 " << month_ << " 월 " << day_
            << " 일 입니다 " << std::endl;
}

int main() {
  Date day = Date();
  Date day2;

  day.ShowDate();
  day2.ShowDate();
  return 0;
}
```

```
오늘은 2019 년 4 월 7 일 입니다 
오늘은 2019 년 4 월 7 일 입니다 
```

Default Constructor는 `Date day = Date()` 혹은 `Date day2`와 같이 인자를 아무것도 주지 않아도 초기화를 시키는 방법이다.

C++ 11부터는 디폴트 생성자를 사용하라고 명시할 수도 있다.

```cpp
class Date {
 public:
  Date() = default;  // 디폴트 생성자를 정의해라
};
```

이후 `Date day1()`을  기입해도 인스턴스를 생성할 수 있다.

<br>

### 생성자 Overloading

```cpp
#include <iostream>

class Date {
  int year_;
  int month_;
  int day_;

 public:
    Date();
    Date(int year, int month, int day);
};

Date::Date() {
    year_ = 2019;
    month_ = 4;
    day_ = 7;
}

Date::Date(int y, int m, int d) {
    year_ = y;
    month_ = m;
    day_ = d;
}

int main() {
  Date day1;
  Date day2 = Date(2019, 4, 8);
  return 0;
}
```

위 코드 블록과 같이, default constructor와 non-default constructor를 동시에 사용할 수 있다.

<br>

### Initializer List

초기화 리스트는 생성자 호출과 동시에 멤버 변수들을 초기화할 때 사용한다.

```cpp
Date::Date() : year_(2019), month_(4), day_7(7) {}
Date::Date(int year, int month, int day) : year_(year), month_(month), day_7(day) {}
```

기존의 방법은 생성자 호출 후 대입을 하는 것이었다면, 초기화 리스트는 생성과 초기화를 동시에 진행하는 방법이다. 이를 변수 생성과 비유하면 다음과 같다.

```cpp
int a = 10; // 초기화 리스트
int a; // 디폴트 생성자 호출
a = 10 // 대입
```

이는 작업을 두 번 시행하는것이며, 이외에도 반드시 초기화 리스트를 사용해야 하는 경우로 레퍼런스를 사용하는 경우가 있다. 만약 **클래스 내부에 레퍼런스 변수 혹은 상수를 넣고 싶다면 반드시 초기화 리스트를 사용해야 한다**.

```cpp
const int a;
a = 3;
```

```cpp
int& ref;
ref = c;
```

위의 두 코드 블록은 각각 상수와 참조를 나누어서 시도하는 경우이며, 모두 에러가 발생한다. 즉, class에서 `const`로 생성하는 멤버 변수가 있을 경우, 일반적인 생성자로는 그 값을 설정할 수 없으며, 반드시 초기화 리스트를 사용해야 한다.

중요하진 않지만, 만약 인자의 이름과 멤버 변수의 이름이 같을 때, 일반 생성자를 사용한다면 컴파일러가 두 가지를 구분할 수 없어 에러가 날 것이고, 초기화 리스트를 사용한다면 인

<br>

### Copy Constructor

복사 생성자는 이미 존재하는 인스턴스의 내용을 복사하여 새로운 인스턴스를 만들어준다. 아래 코드 블록에서 `Date(const Date& d)`와 같이, `d`라는 객체의 주소를 `&`로 참조하여 수행한다. 여기서 `const`를 붙여준 이유는, `d`를 변경하지 않은 채 새로운 인스턴스를 만들기 위함이며, 필수적인 것은 아니다.

```cpp
#include <string.h>
#include <iostream>

class Date {
 public:
    int year_;
    int month_;
    int day_;

    Date(int year, int month, int day);
    Date(const Date& d);
};

Date::Date(int year, int month, int day){
    year_ = year;
    month_ = month;
    day_ = day;
}

Date::Date(const Date& d) {
    std::cout << "복사 생성자 호출 !" << std::endl;
    year_ = d.year_;
    month_ = d.month_;
    day_ = d.day_;
}

int main() {
    Date day1(2019, 4, 7);
    Date day2(day1);
    Date day3 = day2;
    std::cout << day2.day_ << std::endl;
    std::cout << day3.day_ << std::endl;
    return 0;
}
```

```
복사 생성자 호출 !
복사 생성자 호출 !
7
7
```

그런데 여기서 `Date day3 = day2`라고만 해도 복사 생성자가 호출되는 것을 알 수 있다. 복사 생성자는 오직 **생성**하는 작업을 진행할 때, 즉 초기화 할 때에만 사용된다.

그런데 위 코드에서 작성한 복사 생성자를 삭제해도 Default Copy Constructor가 있기 때문에 복사 작업이 잘 이루어진다. 하지만 만약 class 내부에 포인터 변수가 들어갈 경우, 디폴트 복사 생성자는 이를 이용하여 다른 주소의 다른 포인터를 생성하는 것이 아니라, 같은 주소의 변수를 가르키도록 복사(**shallow copy**)한다. 즉, 만약 인스턴스-1을 복사하여 인스턴스-2를 만들고, 인스턴스-1을 삭제할 경우 인스턴스-2의 포인터 변수는 이미 해제된 메모리를 가리키고 있으므로 runtime error가 발생하게 된다.

**_Shallow copy는 대입만 진행해 주며, Deep copy는 매모리를 새로 할당하여 내용을 복사_**

<br>

<br>

## Destructor

만약 인스턴스를 생성했을 때, new를 사용하여 Heap에 인스턴스 변수를 생성하도록 하고 있다면, 인스턴스가 삭제될 때 Heap에 저장된 인스턴스 변수를 삭제하도록 지시해 주어야 한다. 이를 소멸자(destructor)라고 하며, `~ClassName()`으로 지정한다.

```cpp
class Marine {
  char* name;  // 마린 이름

 public:
  Marine(int x, int y, const char* marine_name);  // 이름까지 지정
  ~Marine();
};

Marine::Marine(int x, int y, const char* marine_name) {
  name = new char[strlen(marine_name) + 1];
}

Marine::~Marine() {
  std::cout << name << " 의 소멸자 호출 ! " << std::endl;
  if (name != NULL) {
    delete[] name;
  }
}
```

소멸자는 객체를 `delete`로 삭제시키는 경우 자동으로 호출되며, 프로그램이 종료되어 객체가 자동으로 소멸될 때 또한 호출된다. 소멸자는 아무런 인자를 가질 수 없기 때문에 오버로딩 또한 불가능하다.

<br>

<br>

## Static 변수 초기화

아래 코드에서는 마린을 생성할 때 마다 `static int total_marine_num`의 숫자를 올리고, 마린 인스턴스가 삭제될 때 마다 destructor를 사용하여 값을 낮춰준다.

static int total_marine_num을 초기화 하기 위해 `int Marine::total_marine_num = 0`을 기입했다. 만약 **선언과 동시에 초기화를 하고 싶다면**, `const static int x = 0`과 같은 **const 형태만 가능**하다.

```cpp
#include <iostream>

class Marine {
  // static 변수 선언
  static int total_marine_num;

  int hp;
  int coord_x, coord_y;
  bool is_dead;

  const int default_damage;

 public:
  Marine();
  Marine(int x, int y);
  Marine(int x, int y, int default_damage);
  
  // 인스턴스가 아닌, class로 static 변수 보기
  static void show_total_marine();
    
  ~Marine() { total_marine_num--; }
};

// static 변수 초기화
int Marine::total_marine_num = 0;

// 인스턴스가 아닌, class로 static 변수 보기
void Marine::show_total_marine() {
  std::cout << "전체 마린 수 : " << total_marine_num << std::endl;
}

Marine::Marine()
    : hp(50), coord_x(0), coord_y(0), default_damage(5), is_dead(false) {
  total_marine_num++;
}

Marine::Marine(int x, int y)
    : coord_x(x), coord_y(y), hp(50), default_damage(5), is_dead(false) {
  total_marine_num++;
}

Marine::Marine(int x, int y, int default_damage)
    : coord_x(x),
      coord_y(y),
      hp(50),
      default_damage(default_damage),
      is_dead(false) {
  total_marine_num++;
}

void create_marine() {
  Marine marine3(10, 10, 4);
}

int main() {
  Marine marine1(2, 3, 5);
  Marine marine2(3, 5, 10);
  create_marine();
  // 클래스 통해서 static 변수 확인하기
  Marine::show_total_marine();
}
```

`static` 함수는 앞에서 이야기 한 것과 같이, 어떤 객체에 종속되는 것이 아니라 클래스에 종속되는 것으로, 따라서 이를 호출하는 방법도 `(객체).(멤버 함수)` 가 아니라, `(클래스)::(static 함수)` 형식으로 호출하게 된다.

```cpp
Marine::show_total_marine();
```

<br>

<br>

## this

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

실제로 모든 멤버 함수 내에서는 `this` 키워드가 정의되어 있으며 **클래스 안에서 정의된 함수 중에서 `this` 키워드가 없는 함수는 (당연하게도) `static` 함수 뿐이다.**













