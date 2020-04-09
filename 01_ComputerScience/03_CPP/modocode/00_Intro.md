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

## Namespace

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

## 타입 캐스팅

C++에서 제공하는 캐스팅 네 가지는 다음과 같다.

- `static_cast` : 우리가 흔히 생각하는, 언어적 차원에서 지원하는 일반적인 타입 변환
- `const_cast` : 객체의 상수성(const) 를 없애는 타입 변환
  - e.g., `const int` 자료형을 `int` 로 변환
- `dynamic_cast` : 파생 클래스 사이에서의 다운 캐스팅
- `reinterpret_cast` : 위험을 감수하고 하는 캐스팅으로, 서로 관련이 없는 포인터들 사이의 캐스팅 등

`캐스팅종류<원하는 타입>(캐스팅 대상)`와 같은 형태로 사용할 수 있으며, `float` 자료형을 `int` 자료형으로 바꾸는 것은 다음과 같이 실행한다.

```cpp
static_cast<int>(float_variable);
```

