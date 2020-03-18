# Class Constructor

인스턴스의 생성과 함께, 값을 자동으로 부여하는 것을 constructor라고 합니다.

<br>

<br>

## Automatic Default Constructor

만약 아무런 custom constructor가 정의되어 있지 않다면, C++ **compiler**에 의해 automatic default constructor가 자동으로 작동하게 됩니다. Automatic default constructor는 모든 멤버 변수들을 정의된 default value로 초기화 합니다.

```cpp
// Cube.h
#pragma once

namespace uiuc {
  class Cube {
    public:
    	double getVolume();
    	double getSurfaceArea();
    	void setLength(double length);
    private:
    	double length_; // no default value
  }
};
```

이전 Cube.h에서는 위와 같은 형태로 `Cube` 클래스를 정의했었는데, 이 때도 automatic default constructor는 작동했습니다. 하지만 default value를 따로 지정하지 않았기 때문에 아무런 일이 발생하지 않은 것입니다. 즉, automatic default constructor가 다른 어떤 일도 하지 않습니다만, 멤버 변수를 값을 지정하지 않은 채 메모리 공간에 할당하는 작업을 진행합니다.

<br>

<br>

## Custom Constructor

### Custom Default Constructor

Automatic default constructor는 아무런 값을 할당하지 않았습니다만, 미리 지정해둔 default 값이 있다면 custom default constructor가 해당 default 값을 지정하여 메모리 공간에 할당합니다. 가장 간단한 Custom Default Constructor는 다음과 같이 만들 수 있습니다.

- 클래스 이름과 동일한 이름으로 멤버 함수를 정의한다.

  e.g., `Cube::Cube()`

- 정의한 멤버 함수에는 parameter가 없다.

- 정의한 멤버 함수는 아무것도 반환하지 않는다.

기존에 사용했던 Cube의 헤더 파일과 소스 파일에 `length_` 값을 초기화하는 constructor를 `Cube()`라는 이름으로 정의해 보겠습니다.

```cpp
// Cube.h
#pragma once

namespace uiuc {
  class Cube {
    public:
    	Cube(); // Custom default constructor
    	double getVolume();
    	double getSurfaceArea();
    	void setLength(double length);
    private:
    	double length_;
  }
};
```

```cpp
// Cube.cpp
#include "Cube.h"

namespace uiuc {
  Cube::Cube() {
    length_ = 2;
  }
  ...
}
```

```cpp
// main.cpp
#include "Cube.h"
#include <iostream>

int main() {
  uiuc::Cube c;
  std::cout << "Volume: " << c.getVolume() << std::endl;
  return 0;
}
```

컴파일 후 실행하면, `Volume: 8`이 출력됩니다.

<br>

### Custom Constructor

사용자로부터 정보를 요구하도록 non-default constructor를 정의해 보겠습니다.

```cpp
// Cube.h
#pragma once

namespace uiuc {
  class Cube {
    public:
    	Cube(); // Custom default constructor
    	Cube(double length); // one argument constructor
    	double getVolume();
    	double getSurfaceArea();
    	void setLength(double length);
    private:
    	double length_;
  }
};
```

```cpp
// Cube.cpp
#include "Cube.h"

namespace uiuc {
  Cube::Cube() {
    length_ = 2;
  }
  Cube::Cube(double length) {
    length_ = length;
  }
  ...
}
```

```cpp
// main.cpp
#include "Cube.h"
#include <iostream>

int main() {
  uiuc::Cube c(2);
  std::cout << "Volume: " << c.getVolume() << std::endl;
  return 0;
}
```

컴파일 후 실행하면, `Volume: 8`이 출력됩니다. Custom default constructor와 custom constructor는 동시에 정의되어 있어도 무방합니다.

<br>

### 주의 사항

만약 custom constructor는 정의했지만, custom default constructor가 정의되지 않은 채, 그리고 argument를 부여하지 않은 채 인스턴스를 생성하려 하면 어떻게 될까요?

```cpp
// Cube.h
#pragma once

namespace uiuc {
  class Cube {
    public:
    	Cube(double length); // one argument constructor
    	double getVolume();
    	double getSurfaceArea();
    	void setLength(double length);
    private:
    	double length_;
  }
};
```

```cpp
// Cube.cpp
#include "Cube.h"

namespace uiuc {
  Cube::Cube(double length) {
    length_ = length;
  }
  ...
}
```

```cpp
// main.cpp
#include "Cube.h"
#include <iostream>

int main() {
  uiuc::Cube c; // argument 부여하지 않음
  std::cout << "Volume: " << c.getVolume() << std::endl;
  return 0;
}
```

만약 이런 상태로 compile하려 하면, main.cpp에서 인스턴스를 생성하는 `uiuc::Cube c;` 코드 지점에서  `fatal error: no matching constructor for initialization of 'uiuc::Cube'` 에러가 발생합니다. 또한, class를 선언하는 Cube.h 헤더 파일의 `clss Cube {}`에서 **`copy constructor`** 혹은 **`move constructor`**가 없다는 에러 메세지를 확인할 수 있습니다.

<br>

<br>

## Copy Constructor

이미 존재하는 객체의 복사본을 만드는, C++의 특수한 constructor입니다.

<br>

### Automatic Copy Constructor

만약 사용자가 custom copy constructor를 정의하지 않는다면, compiler가 자동으로 제공하는 constructor입니다. Automatic copy constructor는 **복사하는 객체의 모든 멤버 변수, 그리고 내용(contents)까지 모두 복사**합니다.

Automatic copy constructor의 발동 조건은 다음 두 가지입니다.

1. 복사하려는 객체의 구조가 클래스 구조여야 한다.
2. 단 복사하여 생성하는 클래스가 하나의 argument만을 가져야 하며, argument는 복사하려는 객체와 같은 type의 const reference이다.

`Cube::Cube(const Cube & obj)`

위와 같은 구조로 사용되며, 아래가 그 예시이다.

```cpp
// cpp-cctor/Cube.h
#pragma once

namespace uiuc {
  class Cube {
    public:
      Cube();  // Custom default constructor
      Cube(const Cube & obj);  // Custom copy constructor

      double getVolume();
      double getSurfaceArea();
      void setLength(double length);

    private:
      double length_;
  };
}
```

```cpp
// cpp-cctor/Cube.cpp
#include "Cube.h"
#include <iostream>

namespace uiuc {
  Cube::Cube() {
    length_ = 1;
    std::cout << "Default constructor invoked!" << std::endl;
  }

  Cube::Cube(const Cube & obj) {
    length_ = obj.length_;
    std::cout << "Copy constructor invoked!" << std::endl;
  }
}
```

<br>

### Copy Constructor가 유용한 경우들

Copy Constructor를 유용하게 사용할 수 있는 세 가지 경우들에 대해 예를 들어보겠습니다. 세 가지는 다음과 같습니다.

- 객체를 파라미터로 하여 전달할 경우
- 함수에서 객체를 return하는 경우
- 새 객체를 이전 객체를 기반으로 초기화 할 때

`Cube.h`와 `Cube.cpp` 파일은 위와 동일하게 사용하며, `ex1~3`의 `main.cpp`를 사용하여 각각을 보여드리겠습니다.

```cpp
/* cpp-cctor/ex1/main.cpp
객체를 파라미터로 하여 전달할 경우 */
#include "../Cube.h"
using uiuc::Cube;

void foo(Cube cube) {
  // Nothing :)
}

int main() {
  Cube c;
  foo(c);

  return 0;
}
```

```
Default constructor invoked!
Copy constructor invoked!
```

```cpp
/* cpp-cctor/ex2/main.cpp
함수에서 객체를 return하는 경우 */
#include "../Cube.h"
using uiuc::Cube;

Cube foo() {
  Cube c;
  return c;
}

int main() {
  Cube c2 = foo();
  return 0;
}
```

```
Default constructor invoked!
Copy constructor invoked!
Copy constructor invoked!
```

```cpp
/* cpp-cctor/ex3/main.cpp
새 객체를, 이전 객체를 기반으로 초기화 할 때 */
#include "../Cube.h"
using uiuc::Cube;

int main() {
  Cube c;
  Cube myCube = c;

  return 0;
}
```

```
Default constructor invoked!
Copy constructor invoked!
```

위에서 세 가지 경우에 대한 모든 예시를 들었고, 다음은 **틀린** 예시를 들어보겠습니다. 아래의 경우에는 default constructor만 사용되게 됩니다.

```cpp
/* cpp-cctor/ex4/main.cpp
틀린 예시 */
#include "../Cube.h"
using uiuc::Cube;

int main() {
  Cube c;
  Cube myCube;

  myCube = c;

  return 0;
}
```

```
Default constructor invoked!
Default constructor invoked!
```

위 예시에서는 copy constructor가 발생하는 것이 아니라, default constructor가 2 회 실행된 뒤, 이미 생성된 객체인 `myCube`의 메모리 공간에  `c`라는 객체로 덮어 씌워 버립니다. 즉, 일반적인 변수 할당과 같은 현상이 발생하며, 이전에 생성한 `myCube` 객체의 존재는 사라집니다. 만약  `myCube`라는 객체를 유지한 채,  `=` 연산자를 사용하여,  `c` 라는 객체와 동일한 값도록 하려면, 아래의 Copy Assignment Operator를 사용하면 됩니다.

<br>

<br>

## Copy Assignment Operator

Copy Constructor는 새로운 객체를 생성할 때 사용하는 `constructor`입니다. 반면, assignment Operator는 이미 존재하는 객체에 값을 할당하는 기능을 수행하며, 이미 존재하는 객체일 때에만 사용이 가능합니다.

Assignment operator는 목표하는 객체의 **모든 멤버 변수의 값**들을 지정하는 객체에 할당합니다.

- is a **public member fuction** of the class
- has the function name `operator=`
- has a **return value of a reference** of the class' type
- has exactly **one argument**, and the argument must be const reference of the class' type

위와 같은 규칙들을 갖고 있으며, 아래와 같이 사용할 수 있습니다.

`Cube & Cube::operator=(const Cube & obj)`

구체적인 예시는 다음과 같습니다.

```cpp
// cpp-assignmentOp/main.cpp
#pragma once

namespace uiuc {
  class Cube {
    public:
      Cube();  // Custom default constructor
      Cube(const Cube & obj);  // Custom copy constructor
      
      Cube & operator=(const Cube & obj);  // Custom assignment operator

      double getVolume();
      double getSurfaceArea();
      void setLength(double length);

    private:
      double length_;
  };
}
```

```cpp
// cpp-assignmentOp/Cube.cpp
#include "Cube.h"
#include <iostream>

namespace uiuc {
  Cube::Cube() {
    length_ = 1;
    std::cout << "Default constructor invoked!" << std::endl;
  }

  Cube::Cube(const Cube & obj) {
    length_ = obj.length_;
    std::cout << "Copy constructor invoked!" << std::endl;
  }

  Cube & Cube::operator=(const Cube & obj) {
    length_ = obj.length_;
    std::cout << "Assignment operator invoked!" << std::endl;    
    return *this;
  }
}
```

```cpp
// cpp-assignmentOp/main.cpp
#include "Cube.h"
using uiuc::Cube;

int main() {
  Cube c;
  Cube myCube;

  myCube = c;

  return 0;
}
```

```
Default constructor invoked!
Default constructor invoked!
Assignment operator invoked!
```

Assignment operator가 작동한 것을 확인할 수 있으며, 이로 인해 `myCube`와 `c` 객체 모두 유지한 채, `myCube`의 멤버 변수의 값이 `c` 객체의 값과 동일해 졌음을 알 수 있습니다.



