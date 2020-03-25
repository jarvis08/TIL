# Template Types

Standard library class인 `std::vector`는 "template" type을 이용하여 동적으로 변하는 array를 만들 수 있습니다.

```cpp
// Define
#include <vector>

// Initialize
std::vector<T> v;

// Add to (back) of array
::push_back(T);

// Access specific element
::operator[](unsigned_pos);

// Number of elements
::size()
```

위의 초기화에서, `vector`는 templated type으로 정의되어 있으므로, 원하는 type으로 `T`를 대신하여 사용하면 됩니다. `vector`의 사용 예시는 다음과 같습니다.

```cpp
// cpp-vector/main.cpp
#include <vector>
#include <iostream>

int main() {
  std::vector<int> v;
  for (int i = 0; i < 100; i++) {
    v.push_back( i * i );
  }

  std::cout << v[12] << std::endl;

  return 0;
}
```

```
144
```

<br>

<br>

## Template and Class

Template variable은 class 혹은 function을 선언하기 이전에 선언하여 사용할 수 있습니다. 아래의 예시와 같이, `template <typename T>` 라고 선언한 후, 나머지 내용에서 미리 선언한 `T`를 사용하여 type을 선언하면 됩니다.

```cpp
template <typename T>
class List {
  ...
	private:
  	T data_;
};

template <typename T>
int max(T a, T b) {
	if (a > b) { return a; }
  return b;
}
```

Template variable들은 컴파일 할 때 확인되므로, 개발자가 실수를 했을 때 에러가 발생하여 알려줍니다.

<br>

### Example

```cpp
// cpp-templates/Cube.h
#pragma once
#include <iostream>

namespace uiuc {
  class Cube {
    public:
      Cube(double length);  // One argument constructor
      Cube(const Cube & obj);  // Custom copy constructor

      Cube & operator=(const Cube & obj);  // Custom assignment operator

      double getVolume() const;
      double getSurfaceArea() const;
      void setLength(double length);

      // An overloaded operator<<, allowing us to print the Cube via `cout<<`:
      friend std::ostream& operator<<(std::ostream & os, const Cube & cube);

    private:
      double length_;
  };
}
```

```cpp
// cpp-templates/Cube.cpp
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

  std::ostream& operator<<(std::ostream & os, const Cube & cube) {
    os << "Cube(" << cube.length_ << ")";
    return os;
  }
}
```

```cpp
// cpp-templates/main.cpp
#include <iostream>
using std::cout;
using std::endl;

#include "Cube.h"
using uiuc::Cube;

template <typename T>
T max(T a, T b) {
  if (a > b) { return a; }
  return b;
}

int main() {
  cout << "max(3, 5): " << max(3, 5) << endl;
  cout << "max('a', 'd'): " << max('a', 'd') << endl;
  cout << "max(\"Hello\", \"World\"): " << max("Hello", "World") << endl;
  // cout << "max( Cube(3), Cube(6) ): " << max( Cube(3), Cube(6) ) << endl; 에러 발생

  return 0;
}
```

```
max(3, 5): 5
max('a', 'd'): d
max("Hello", "World"): World
```



