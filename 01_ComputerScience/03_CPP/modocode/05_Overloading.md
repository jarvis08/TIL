# Overloading

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
   - `포인터는 `void` 포인터로 변환된다.
4. 유저 정의된 타입 변환으로 일치하는 것을 찾는다.

<br>

<br>

## 연산자 오버로딩

`(반환 타입) operator(연산자)(인자)`

`(반환 타입) (클래스명)::operator(연산자)(인자)`

<br>

### 대입(`=`) 연산자

사실은 굳이 `operator=` 를 만들지 않더라도, 위 소스를 컴파일 하면 잘 작동한다. 이는 컴파일러 차원에서 디폴트 대입 연산자(default assignment operator)를 지원하고 있기 때문이지만, 디폴트 복사 생성자와 마찬가지로 디폴트 대입 연산자 역시 **얕은 복사**를 수행한다. 따라서, 깊은 복사가 필요한 클래스의 경우 (예를 들어, 클래스 내부적으로 동적으로 할당되는 메모리를 관리하는 포인터가 있다던지) 대입 연산자 함수를 꼭 만들어주어야 할 필요가 있다.

```cpp
#include <iostream>

class Complex {
 private:
  double real, img;

 public:
  Complex(double real, double img) : real(real), img(img) {}
  Complex(const Complex& c) { real = c.real, img = c.img; }

  Complex operator+(const Complex& c);
  Complex operator-(const Complex& c);
  Complex operator*(const Complex& c);
  Complex operator/(const Complex& c);

  Complex& operator=(const Complex& c);
  void println() { std::cout << "( " << real << " , " << img << " ) " << std::endl; }
};

// 연산자 오버로딩
Complex Complex::operator+(const Complex& c) {
  Complex temp(real + c.real, img + c.img);
  return temp;
}
Complex Complex::operator-(const Complex& c) {
  Complex temp(real - c.real, img - c.img);
  return temp;
}
Complex Complex::operator*(const Complex& c) {
  Complex temp(real * c.real - img * c.img, real * c.img + img * c.real);
  return temp;
}
Complex Complex::operator/(const Complex& c) {
  Complex temp(
    (real * c.real + img * c.img) / (c.real * c.real + c.img * c.img),
    (img * c.real - real * c.img) / (c.real * c.real + c.img * c.img));
  return temp;
}

// 대입 연산자
Complex& Complex::operator=(const Complex& c) {
  real = c.real;
  img = c.img;
  return *this;
}

int main() {
  Complex a(1.0, 2.0);
  Complex b(3.0, -2.0);
  Complex c(0.0, 0.0);
  c = a * b + a / b + a + b;
  // 대입 연산자 사용
  c.println();
}
```

대입 연산자는 복사 생성자와 유사하지만 다른 역할을 수행한다.

```cpp
// 복사 생성자
Some_Class a = b;

// 초기화 후 대입 연산자 사용
Some_Class a;
a = b;
```

복사 생성자는 초기화 시 함께 사용되어야 하며, 만약 코드의 아래부분과 같이 작성 시, 초기화 후 대입 연산자를 활용한 것이다. 즉, 두 코드는 유사한 기능이지만, 분명 다른 과정으로 `a`의 값을 조정한다.

<br>

<br>

## Wrapper Class

> `Wrapper` 라는 것은 원래 우리가 흔히 음식을 포장할 때 '랩(wrap)으로 싼다' 라고 하는 것 처럼, '포장지' 라는 의미의 단어 입니다. 즉 `Wrapper` 클래스는 무언가를 포장하는 클래스라는 의미인데, C++ 에서 프로그래밍을 할 때 어떤 경우에 기본 자료형들을 객체로써 다루어야 할 때가 있습니다. 이럴 때, 기본 자료형들 (`int, float` 등등) 을 클래스로 포장해서 각각의 자료형을 객체로 사용하는 것을 `Wrapper` 클래스를 이용한다는 뜻 입니다.

`int` 자료형의 wrapper 클래스를 만들어 보면 다음과 같다.

```cpp
class Int
{
  int data;
  // some other data

 public:
  Int(int data) : data(data) {}
  Int(const Int& i) : data(i.data) {}
};
```

<br>

### 타입 변환 연산자

`operator (변환 하고자 하는 타입) ()`

> 그렇다면, 그냥 이 `Wrapper` 클래스의 객체를 마치 '`int` 형 변수' 라고 컴파일러가 생각할 수 는 없는 것일까요. 물론 가능합니다. 왜냐하면 우리에게는 타입 변환 연산자가 있기 때문이지요. 만일 컴파일러가 이 클래스의 객체를 `int` 형 변수로 변환할 수 있다면, 비록 `operator+` 등을 정의하지 않더라도 컴파일러가 가장 이 객체를 `int` 형 변수로 변환 한 다음에 `+` 를 수행할 수 있기 때문입니다.

```cpp
#include <iostream>

class Int {
  int data;
  // some other data

 public:
  Int(int data) : data(data) {}
  Int(const Int& i) : data(i.data) {}
	
  // 타입 변환 연산자
  operator int() { return data; }
};

int main() {
  Int x = 3;
  // 타입 변환 연산자로 인해 컴파일러가 x를 int로 취급
  int a = x + 4;

  // 타입 변환 연산자로 인해 컴파일러가 x를 int로 취급
  x = a * 2 + x + 4;
  std::cout << x << std::endl;
}
```

<br>

### 전위/후위 증감 연산자

```cpp
// 전위 증감 연산자
operator++();
operator--();

// 후위 증감 연산자
operator++(int x);
operator--(int x);

// 후위 증감 연산자
operator++(int);
operator--(int);
```

 인자 `x` 는 아무런 의미가 없으며, 단순히 컴파일러 상에서 전위와 후위를 구별하기 위해 `int` 인자를 넣어준다. 또한, 실제로 `++` 을 구현하면서 인자로 들어가는 값을 사용하는 경우는 없으므로 `x`를 제거해줄 수 있다.

 하지만 중요한 것은, 증감 연산자는 자기 자신을 return해야 한다. 전위의 경우 값이 바뀐 자신을, 후위의 경우 값이 바뀌기 전을 return 해야 한다.

```cpp
A& operator++() {
  // A ++ 을 수행한다.
  return *this;
}

A operator++(int) {
  A temp(A);
  // A++ 을 수행한다.
  return temp;
}
```

`++` 을 하기 전에 객체를 반환해야 하므로, `temp` 객체를 만들어서 이전 상태를 기록한 후에, `++` 을 수행한 뒤에 `temp` 객체를 반환하게 됩니다. 따라서 후위 증감 연산의 경우 추가적으로 복사 생성자를 호출하기 때문에 전위 증감 연산보다 더 느리다.

```cpp
#include <iostream>


class Test {
  int x;

 public:
  Test(int x) : x(x) {}
  Test(const Test& t) : x(t.x) {}

  Test& operator++() {
    x++;
    std::cout << "전위 증감 연산자" << std::endl;
    return *this;
  }

  // 전위 증감과 후위 증감에 차이를 두기 위해 후위 증감의 경우 인자로 int 를
  // 받지만 실제로는 아무것도 전달되지 않는다.
  Test operator++(int) {
    Test temp(*this);
    x++;
    std::cout << "후위 증감 연산자" << std::endl;
    return temp;
  }

  int get_x() const {
    return x;
  }
};

void func(const Test& t) {
  std::cout << "x : " << t.get_x() << std::endl;
}

int main() {
  Test t(3);

  func(++t); // 4
  func(t++); // 4 가 출력됨
  std::cout << "x : " << t.get_x() << std::endl;
}
```

```
전위 증감 연산자
x : 4
후위 증감 연산자
x : 4
x : 5
```
