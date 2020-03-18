# Class Destructor

해당 클래스의 인스턴스들을 삭제하는 기능을 합니다.

<br>

### Automatic Default Destructor

만약 따로 destructor를 작성해주지 않는다면, 자동적으로 automatic default destructor를 사용합니다. 하지만, automatic default destructor는 멤버 변수들을 초기화하는 기능만 있습니다. 따라서 만약 logging을 하거나 추가적인 메모리 cleanup을 진행하려면, custom destructor를 작성해야 합니다.

Automatic default destructor는 절대 직접적으로 호출될 수 없으며, 객체의 메모리가 시스템으로 reclaimed될 때 호출됩니다.

- 객체가 stack에 저장됐을 경우, 함수가 `return`을 진행할 때 호출
- 객체가 heap에 저장됐을 경우, `delete`가 사용될 때 호출

즉, automatic default destructor는 compile 시에 호출되는 것이 아니라, runtime 동안에 호출되게 됩니다. Compiler는 암시적으로 특정 공간에 desctructor를 준비시킨 후, runtime 동안 조건 만족 시 사용하게 됩니다.

<br>

### Custom Destructor

Custom destructor는 다음과 같은 성질을 가집니다.

- Custom destructor는 멤버 함수이다.
- tild(`~`)로 시작하는 클래스 이름으로 정의한다.
- 모든 destructor들은 argument가 없으며, 아무것도 반환하지 않는다.

아래와 같이 사용할 수 있습니다.

`Cube::~Cube()`

아래 에시를 통해 stack과 heap에서 생성한 인스턴스들을 삭제해 보겠습니다.

```cpp
// cpp-dtor/Cube.h
#pragma once

namespace uiuc {
  class Cube {
    public:
      Cube();  // Custom default constructor
      Cube(double length);  // One argument constructor
      Cube(const Cube & obj);  // Custom copy constructor
      ~Cube();  // Destructor

      Cube & operator=(const Cube & obj);  // Custom assignment operator

      double getVolume() const;
      double getSurfaceArea() const;
      void setLength(double length);

    private:
      double length_;
  };
}
```

```cpp
// cpp-dtor/Cube.cpp
#include "Cube.h"
#include <iostream>

using std::cout;
using std::endl;

namespace uiuc {  
  Cube::Cube() {
    length_ = 1;
    cout << "Created $1 (default)" << endl;
  }

  Cube::Cube(double length) {
    length_ = length;
    cout << "Created $" << getVolume() << endl;
  }

  Cube::Cube(const Cube & obj) {
    length_ = obj.length_;
    cout << "Created $" << getVolume() << " via copy" << endl;
  }

  Cube::~Cube() {
    cout << "Destroyed $" << getVolume() << endl;
  }

  Cube & Cube::operator=(const Cube & obj) {
    cout << "Transformed $" << getVolume() << "-> $" << obj.getVolume() << endl;
    length_ = obj.length_;
    return *this;
  }



  double Cube::getVolume() const {
    return length_ * length_ * length_;
  }

  double Cube::getSurfaceArea() const {
    return 6 * length_ * length_;
  }

  void Cube::setLength(double length) {
    length_ = length;
  }
}
```

```cpp
// cpp-dtor/main.cpp
#include "Cube.h"
using uiuc::Cube;

double cube_on_stack() {
  Cube c(3);
  return c.getVolume();
}

void cube_on_heap() {
  Cube * c1 = new Cube(10);
  Cube * c2 = new Cube;
  delete c1;
}

int main() {
  cube_on_stack();
  cube_on_heap();
  cube_on_stack();
  return 0;
}
```

```bash
# stack
Created $27
Destroyed $27

# heap
Created $1000
Created $1 (default)
# c1이 삭제됐으며, c2는 삭제되지 않았다.
Destroyed $1000

# stack
Created $27
Destroyed $27
```

위 예제를 보면 heap 메모리에 생성된 두 개의 인스턴스 중 하나의 인스턴스만이 삭제된 것을 알 수 있습니다. 위의 heap과 같이, 아래의 내용들은 반드시 삭제하도록 지정해 주어야 합니다.

- Heap memory
- Opened files
- Shared memory