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
