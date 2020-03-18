# Storage of Variable

모든 변수들 및 인스턴스들은 다음 세 가지 방법으로 저장될 수 있습니다.

- Directly in memory
- Accessed via a pointer
- Accessed by a reference

<br>

### Direct Storage

By default, 변수들은 메모리에 direct하게 저장됩니다. 변수의 type은 수정될 수 없으며, 객체는 정확히 그 크기만큼 메모리 공간을 차지합니다.

```cpp
Cube c; // Stores a Cube in memory
int i; // Stores an integer in memory
uiuc::HSLAPixel p; // Stores a pixel in memory
```

<br>

### Storage by Pointer

포인터의 type은 asterisk(`*`)로 변경될 수 있습니다. 포인터는 '메모리 주소 폭' 만큼의 공간을 차지하며, 64-bit system의 경우 64 bits를 차지합니다. 포인터는 객체가 할당된 공간을 가리킵니다.

```cpp
Cube *c; // Points to a Cube in memory
int *i; // Points to an integer in memory
uiuc::HSLAPixel *p; // Points to a pixel in memory
```

<br>

### Storage by a Reference

Reference를 사용하여 이미 존재하는 메모리를 alias 하는 것이며, ampersand(`&`)를 사용하여 이미 존재하는 메모리의 type을 유지한 채 사용할 수 있습니다. Reference는 스스로 메모리 공간을 갖고 있지 않으며, 그저 또 다른 변수를 alias 하기만 합니다. 

이를 위해서는 **해당 변수가 초기화될 때 선언되어야만 합니다.**

```cpp
Cube &c = cube; // Alias to the variabel 'cube'
int &i = count; // Alias to the variable 'count'
uiuc::HSLAPixel &p; // Illegal! Must alias sth when variable is initialized
```

아래에서는 다음과 같은 작업들에 대한 예시를 들어보겠습니다.

- 값 전달하기
- 포인터로 전달하기
- 참조로 전달하기

```cpp
// cpp-memory2/Cube.h
#pragma once

namespace uiuc {
  class Cube {
    public:
      Cube(double length);  // One argument constructor
      Cube(const Cube & obj);  // Custom copy constructor

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
// // cpp-memory2/Cube.cpp
#include "Cube.h"
#include <iostream>

namespace uiuc {  
  Cube::Cube(double length) {
    length_ = length;
    std::cout << "Created $" << getVolume() << std::endl;
  }

  Cube::Cube(const Cube & obj) {
    length_ = obj.length_;
    std::cout << "Created $" << getVolume() << " via copy" << std::endl;
  }

  Cube & Cube::operator=(const Cube & obj) {
    std::cout << "Transformed $" << getVolume() << "-> $" << obj.getVolume() << std::endl;
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
/* cpp-memory2/ex1/byValue
값으로 전달하기 */
#include "../Cube.h"
using uiuc::Cube;

int main() {
  // Create a 1,000-valued cube
  Cube c(10);

  // Transfer the cube
  Cube myCube = c;

  return 0;
}
```

```
Created $1000
Created $1000 via copy
```

```cpp
/* cpp-memory2/ex1/byPointer
포인터로 전달하기 */

#include "../Cube.h"
using uiuc::Cube;

int main() {
  // Create a 1,000-valued cube
  Cube c(10);

  // Transfer the cube
  Cube * myCube = &c;

  return 0;
}
```

```
Created $1000
```

```cpp
/* cpp-memory2/ex1/byRef
참조로 전달하기 */
#include "../Cube.h"
using uiuc::Cube;

int main() {
  // Create a 1,000-valued cube
  Cube c(10);

  // Transfer the cube
  Cube & myCube = c;

  return 0;
}
```

```
Created $1000
```

<br>

### 데이터 전달하기

위와 같이, 데이터는 다음과 같이 세 가지 방법으로 전달될 수 있습니다.

- Pass by value (default)
- Pass by pointer (modified with `*`)
- Pass by reference (modified with `&`, acts an alias)
  - 또 다른 변수를 생성하지만, 같은 메모리 주소를 바라보도록 함

위에서 설명했던 `operator=`에 대한 예시로는 포인터와 참조의 차이가 없습니다. 아래 예시에서는 같은 헤더 파일과 `Cube.cpp`를 사용하지만, 다른 결과를 보여줄 수 있도록 함수(`sendCube`)를 사용해보겠습니다. 함수를 통해 객체를 전달했을 때, 객체의 복사가 이루어지는지, 혹은 주소를 참조만 하여 복사가 발생하지 않는지를 확인해 보겠습니다.

```cpp
/* cpp-memory2/ex2/byValue
값으로 전달하기 */
#include "../Cube.h"
using uiuc::Cube;

bool sendCube(Cube c) {    
  // ... logic to send a Cube somewhere ...
  return true;
}

int main() {
  // Create a 1,000-valued cube
  Cube c(10);

  // Send the cube to someone
  sendCube(c);

  return 0;
}
```

```
Created $1000
Created $1000 via copy
```

```cpp
/* cpp-memory2/ex2/byPointer
포인터로 전달하기 */
#include "../Cube.h"
#include <iostream>

using uiuc::Cube;

bool sendCube(Cube * c) {    
  // ... logic to send a Cube somewhere ...
  return true;
}

int main() {
  // Create a 1,000-valued cube
  Cube c(10);

  // Send the cube to someone
  sendCube(&c);

  return 0;
}
```

```
Created $1000
Created $1000 via copy
```

```cpp
/* cpp-memory2/ex2/byRef
참조로 전달하기 */
#include "../Cube.h"
#include <iostream>

using uiuc::Cube;

bool sendCube(Cube & c) {    
  // ... logic to send a Cube somewhere ...
  return true;
}

int main() {
  // Create a 1,000-valued cube
  Cube c(10);

  // Send the cube to someone
  sendCube(c);

  return 0;
}
```

```
Created $1000
```

값과 포인터를 사용했을 때에는 copy constructor가 작동하지만, reference를 사용했을 때에는 주소만을 옮기므로, 원본에만 작업이 이루어지는 것을 알 수 있습니다.

그런데 reference를 사용할 때 주의해야 할 점은, **현재 함수의 stack에서 생성된  stack variable의 reference를 return하면 안됩니다.**













