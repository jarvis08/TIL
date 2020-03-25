# Inheritance

Base class로 부터 모든 멤버 함수와 변수를 상속받을 수 있습니다. 상속은 거의 모든 경우에 `public`으로 이루어집니다. 만약 `Cube` class가 `Shape` class를 상속받는다면, `Cube` class 내부에서 `Shape` class를 `public`으로 선언하며, `Cube` class가 초기화 될 때 `Shape` class가 (반드시) 함께 초기화됩니다.

```cpp
// cpp-inheritance/Shape.h
#pragma once

class Shape {
  public:
    Shape();
    Shape(double width);
    double getWidth() const;

  private:
    double width_;
};
```

```cpp
// cpp-inheritance/Cube.h
#pragma once

#include "Shape.h"
#include "HSLAPixel.h"

namespace uiuc {
  class Cube : public Shape {
    public:
      Cube(double width, uiuc::HSLAPixel color);
      double getVolume() const;

    private:
      uiuc::HSLAPixel color_;
  };
}
```

<br>

## Initialization List

만약 **initialization list**를 활용한다면, `Shape` class 초기화 시 custom constructor를 사용할 수 있습니다. 아래 `Cube.cpp`에서는 `Shape(width)`를 통해 width를 지정하여 custom constructor를 사용합니다.

```cpp
// cpp-inheritance/Cube.cpp
#include "Cube.h"
#include "Shape.h"

namespace uiuc {
  Cube::Cube(double width, uiuc::HSLAPixel color) : Shape(width) {
    color_ = color;
  }

  double Cube::getVolume() const {
    // Cannot access Shape::width_ due to it being `private`
    // ...instead we use the public Shape::getWidth(), a public function

    return getWidth() * getWidth() * getWidth();
  }
}
```

<br>

### Access Control

Base class가 상속됐을 때, derived class는 다음 권한을 갖습니다.

- Base class의 모든 `public` member들에 접근
- Base class의 `private` member들에는 접근 불가

즉, 위의 `Cube::getVolume()` 멤버 함수 내에서 사용된 `getWidth()` 함수는 `Shape`의 멤버 함수 이지만, `Shape` class를 상속받은 `Cube` class에서 사용하는것이 가능합니다.

<br>

### Initializer List

Base class를 초기화하는데 사용되는 initializer를 initializer list라고 부르며, 다음과 같은 목적으로 사용될 수 있습니다.

- Base class를 초기화
- 다른 class(base class) constructor를 사용하여 현재의(derived) class를 초기화
- (derived class's)멤버 변수를 (base classes')default value들로 초기화

```cpp
// cpp-inheritance/Shape.cpp
#include "Shape.h"

Shape::Shape() : Shape(1) {
  // Nothing.
}

Shape::Shape(double width) : width_(width) {
  // Nothing.
}

double Shape::getWidth() const {
  return width_;
}
```

위 `Shape.cpp`에서는 `Shape(1)`로 인해 1의 길이로 초기화 하는 기능이 있습니다.

```cpp
// cpp-inheritance/main.cpp
#include <iostream>

#include "Cube.h"
#include "HSLAPixel.h"

int main() {
  uiuc::Cube c(4, uiuc::HSLAPixel::PURPLE);
  std::cout << "Created a Purple cube!" << std::endl;
  return 0;
}
```

```
Created a Purple cube!
```

