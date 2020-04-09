# Constructor & Destructor

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

이는 작업을 두 번 시행하는것이며, 이외에도 반드시 초기화 리스트를 사용해야 하는 경우로 레퍼런스를 사용하는 경우가 있다. 만약 클래스 내부에 **레퍼런스 변수** 혹은 **상수**를 넣고 싶다면 **반드시 초기화 리스트를 사용**해야 한다.

```cpp
const int a;
a = 3;
```

```cpp
int& ref;
ref = c;
```

위의 두 코드 블록은 각각 상수와 참조를 나누어서 시도하는 경우이며, 모두 에러가 발생한다. 즉, class에서 `const`로 생성하는 멤버 변수가 있을 경우, 일반적인 생성자로는 그 값을 설정할 수 없으며, 반드시 초기화 리스트를 사용해야 한다.

또한, 만약 인자의 이름과 멤버 변수의 이름이 같을 때, 일반 생성자를 사용한다면 컴파일러가 두 가지를 구분할 수 없어 에러가 날 것이다.

<br>

<br>

## Copy Constructor

복사 생성자는 이미 존재하는 인스턴스의 내용을 복사하여 새로운 인스턴스를 만들어준다. 아래 코드 블록에서 **`Date(const Date& d)`와 같이, `d`라는 객체의 주소를 `&`로 참조**하여 수행한다. 여기서 `const`를 붙여준 이유는, `d`를 변경하지 않은 채 새로운 인스턴스를 만들기 위함이며, 필수적인 것은 아니다.

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

### `explicit` 생성자

다음과 같은 코드가 있을 때, MyString이라는 클래스는 int capacity라는 인자를 받는 생성자가 있으므로 `MyString s(5)` 뿐만 아니라,  `MyString s = 5;`와 같은 형태로도 인스턴스 생성이 가능하다. 이는 컴파일러가 암시적 변환(implicit)을 진행했기 때문이며, 에러가 발생하지는 않지만 차후에 에러를 유발하는 원인이 될 수 있다. 이런 경우를 막기 위해 `explicit`을 선언해 주면 후자의 경우를 막을 수 있다.

```cpp
#include <iostream>

class MyString {
  char* string_content;  // 문자열 데이터를 가리키는 포인터
  int string_length;     // 문자열 길이
  int memory_capacity;
 public:
  // explicit을 이용하여 capacity 만큼 미리 할당함
  explicit MyString(int capacity);
    
  // 문자열로 부터 생성
  MyString(const char* str);
  // 복사 생성자
  MyString(const MyString& str);
  ~MyString();
};

int main() {
  MyString s(5);
  MyString s = 5; // 에러 발생
}
```

<br>

### Static 변수 초기화

 클래스의 static 멤버변수는 클래스 내에 **선언하는 것만으로는 공간이 할당되지 않는다**. 클래스 내에 선언된 형태를 바탕으로, 클래스 밖에서 변수를 선언해야 메모리 공간을 할당받고 변수를 사용할 수 있게 된다. 또한 클래스의 static 멤버 변수는 클래스 객체의 생성시 클래스의 일부분으로서 저장되는 것이 아니라, 별도로 저장이 된다.

 `static int total_marine_num`을 초기화 하기 위해 `int Marine::total_marine_num = 0`을 기입했다. 만약 **선언과 동시에 초기화를 하고 싶다면**, `const static int x = 0`과 같은 **const 형태만 가능**하다.

```cpp
#include <iostream>

class Marine {
  // static 변수 선언
  static int total_marine_num;
  ...
  
  // 인스턴스가 아닌, class로 static 변수 보기
  static void show_total_marine();
  
  // 소멸자로 static의 marine count 낮추기
  ~Marine() { total_marine_num--; }
};

// static 변수 초기화
int Marine::total_marine_num = 0;

// 인스턴스가 아닌, class로 static 변수 보기
void Marine::show_total_marine() {
  std::cout << "전체 마린 수 : " << total_marine_num << std::endl;
}

int main() {
  Marine marine(2, 3, 5);
  // 클래스 통해서 static 변수 확인하기
  Marine::show_total_marine();
}
```

아래 코드에서는 마린을 생성할 때 마다 `static int total_marine_num`의 숫자를 올리고, 마린 인스턴스가 삭제될 때 마다 destructor를 사용하여 값을 낮춰준다.

`static` 함수는 앞에서 이야기 한 것과 같이, 어떤 객체에 종속되는 것이 아니라 클래스에 종속되는 것으로, 따라서 이를 호출하는 방법도 `(객체).(멤버 함수)` 가 아니라, `(클래스)::(static 함수)` 형식으로 호출하게 된다.

```cpp
Marine::show_total_marine();
```

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