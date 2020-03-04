# Memory

### 0-1. 변수의 구성

- type
- name
- value
- memory address
  - `&name` 형태로 열람 가능
  - `int` type in 64-bit system: 64 bit(8 byte)의 공간 할당
  - `int` type in 32-bit system: 32 bit(4 byte) 공간 할당
  - Visual Studio에서는 시각적으로 메모리 확인 가능
    - `int var = 8;` in 32-bit system: `08 00 00 00`
    - `int var = 8;` in 64-bit system: `08 00 00 00 00 00 00 00`

<br>

## 1. Reference

Reference는 `&`로 표현하며, 해당 변수의 주소를 나타내는데 사용합니다. 이는 매우 유용한 기능이며, 아래에서 설명할 포인터는 물론이며, 함수 호출 등에서도 매우 자주 사용됩니다. 함수 호출을 할 때 마다 배열을 모두 옮겨다니는 것은 매우 비효율적이며, 주소만을 전달하여 직접 수정하는 것이 효율적입니다.

<br>

### 1-1. 참조형 변수

```cpp
#include <iostream>
int main() {
  int value = 5; // normal integer
  int& ref = value; // reference to variable value
  value = 6; // value is now 6
  ref = 7; // value is now 7
  std::cout << value; // prints 7
  ++ref; std::cout << value; // prints 8
  return 0;
}
출처: https://boycoding.tistory.com/207 [소년코딩]
```

<br>

<br>

## 2. Stack Memory

C++에서 모든 변수가 저장되는 default 공간입니다. 현재 작동하고 있는 function 또한 스택 공간에 그 내용이 저장되며, 해당 공간은 함수의 종료와 함께 시스템으로 반환됩니다.

스택 메모리는 언제나 높은 주소로 부터 시작하여, 낮아지는 방향으로 저장합니다. 이러한 이유로 인해 메모리를 읽을 때에도 큰 주소에서 작은 주소로 읽게 됩니다. 만약 메모리에 `08 12 22 31`과 같은 형태로 숫자가 저장되어 있을 때, 이는 실제로 `31221208`을 의미합니다.

그런데 헷갈리지 말아야 할 것은, 아래에서 사용하게 될 pointer와 reference는 가장 낮은 주소를 저장하게 됩니다. 즉, 똑같이 `08 12 22 31` 값들이 메모리 주소에 저장되어 있을 때, pointer와 reference가 가리키는 주소는 `08`의 주소이며, 읽는 것은 `31` 부터 읽습니다.

<br>

<br>

## 3. Pointer

다른 변수의 주소를 저장하는 변수입니다. 기본적인 사용법은 아래와 같습니다.

```cpp
int num = 8;
int * ptr = &num;
```

위와 같이 `ptr` 이라는 포인터 변수에 `num` 변수의 주소를 저장한 후 출력값은 다음과 같습니다.

- `std::cout << ptr << std::endl;` : `num` 변수의 의 주소
- `std::cout << *ptr << std::endl;` : 8

<br>

### 3-1. 포인터가 가리키는 주소의 데이터 값을 변경하기

아래와 같이 실제로 `num` 변수에 저장된 값을 변경할 수 있습니다.

```cpp
*ptr = 10;
```

변경 후, `std::cout << num`을 통해 변경된 값인 10이 출력됨을 확인해 볼 수 있습니다.

<br>

여기서 `int * ptr = &num;`라고 선언하며 `int`임을 지정했는데, 이는 `num` 변수의 메모리 공간이 `int` 임을 알리기 위함이지, `ptr` 포인터 자체에는 아무런 영향을 미치지 않습니다. 만약 포인터로 아무런 작업을 하지 않으며, 단지 메모리 주소만을 알고 싶다면, `void *ptr` 혹은  `char *ptr` 모두 아무런 상관이 없습니다. `std::out << ptr;` 라고 출력해 보면, 모두 동일한 주소 값을 반환합니다. 

하지만, `*ptr = 15;`와 같이 주소를 통해 해당 메모리의 값을 변경하고자 한다면, 이 값이 어떤 자료형인지 알 필요가 있습니다. 자료형을 맞춰주지 않은 채 실행하면 오류가 발생합니다. 따라서  `num`과 `ptr`의 type을 맞춰주어야 합니다.

<br>

### 3-2. 주소의 주소를 저장하기

포인터의 주소를 또 다른 포인터에 저장하는 것도 가능합니다.

```cpp
int num = 8;
int * ptr = &num;
void * p = &ptr;
```

위와 같이 작성한다면, `ptr` 포인터는 `num`의 주소를 저장하며, `p` 포인터는 `ptr` 포인터의 주소를 저장합니다. 만약 `p` 포인터를 사용하여 `num` 값을 변경하고 싶다면, `**p = 10`과 같이 asterisk(`*`)를 두 개 사용하면 됩니다.

<br>

### 3-3. 주의 사항

함수 혹은 멤버 함수를 포인터로 사용하는 것에는 주의가 필요합니다. 예를 들어 설명하자면, 다음과 같습니다.

```cpp
// cpp-memory/puzzle.cpp
#include <iostream>
#include "Cube.h"
using uiuc::Cube;

Cube *CreateUnitCube() {
  Cube cube;
  cube.setLength(15);
  return &cube;
}

int main() {
  Cube *c = CreateUnitCube();
  someOtherFunction();
  double a = c->getSurfaceArea();
  std::cout << "Surface Area: " << a << std::endl;
  double v = c->getVolume();
  std::cout << "Volume: " << v << std::endl;
  return 0;
}
```

위와 같이 작성된 코드가 있습니다. `main()` 함수를 보면, `Cube`의 인스턴스로 `*c` 포인터를 지정했습니다. 그렇게 한 이유는 `CreateUnitCube()`를 통해 `Cube`의 인스턴스를 생성하고 초기값을 지정한 뒤, 생성된 인스턴스 `cube`의 주소를 반환하고자 했기 때문입니다.

하지만 `main()` 함수 내에서 면적을 구하고 부피를 구한 뒤, `cout`을 통해 출력해 보지만 모두 0의 값이 나옵니다. 이러한 결과가 나온 것은 stack 메모리의 특성 때문입니다. Stack 메모리는 **함수를 실행할 때 메모리 공간에 등록되고, 종료 시 사용한 메모리를 모두 시스템에 반환**합니다. 따라서 함수 내에서 생성한 `Cube`의 인스턴스인 `cube`의 메모리 또한 시스템에 반환되었고, `main()` 함수에서는 `cube` 인스턴스가 아니라 빈 공간을 호출하고 작동시킨 셈입니다.
