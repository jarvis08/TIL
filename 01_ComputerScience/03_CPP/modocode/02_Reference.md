# Reference

레퍼런스는 어떤 변수를 참고하는, 다른 이름을 부여하는 것과 같다.

```c++
int main() {
    int number = 1;
    int& ref = number;
    ref = 3;
}
```

을 작성할 경우, `number`에는 3의 값이 저장됩니다. 주의할 것은 reference는 변경될 수 없다는 점이다. 아래와 같이 실행할 경우, `b`의 값이 `a`에 저장되는 것이며, `a`의 주소와 `b`의 주소가 같아지는 것이 아니다.

```c++
int a = 10;
int &ref = a;
int b = 3;
ref = b;
```

Reference와 pointer의 차이는 아래를 통해 확인할 수 있다.

```c++
#include <iostream>

int main() {
    int number = 10;
    int& ref = number;
    int* p = &number;
    int* p2 = &number;
    for (int i=0; i<5; i++){
        ref++;
        (*p)++;
        *p = *p + 1;
        p2++;
        std::cout << number << std::endl;
        std::cout << p << std::endl;
        std::cout << p2 << std::endl;
    }
}
```

```
13
0x7ffc7f602458
0x7ffc7f60245c
16
0x7ffc7f602458
0x7ffc7f602460
19
0x7ffc7f602458
0x7ffc7f602464
22
0x7ffc7f602458
0x7ffc7f602468
25
0x7ffc7f602458
0x7ffc7f60246c
```

실행 결과, `ref++`에 의해 `number`는 1이 추가될 것이며, `p2++`의 경우 `p2`가 이상한 주소를 가리키게 한다.

<br>

### Reference의 reference

`number`를 참조하는 `&ref1`과, `&ref1`을 참조하는 `ref2`가 있을 때 포인터와의 차이를 알 수 있다.

```cpp
#include <iostream>

int main() {
    int number = 3;
    // reference 이용
    int& ref1 = number;
    int& ref2 = ref1;
    
    // pointer 이용
    int* ptr1 = &number;
    int** ptr2 = &ptr1;
}
```

위와 같이 사용해야 모든 reference와 pointer들이 `number` 변수를 가리키도록 할 수 있다.

<br>

### 함수 내에서, 함수 외의 변수 수정

```c++
#include <iostream>

int by_ref(int &p) {
  p = 3;
  return 0;
}

int by_ptr(int *p) {
  *p = 3;
  return 0;
}

int main() {
  int ref_num = 5;
  int ptr_num = 5;

  std::cout << ref_num << std::endl;
  std::cout << ptr_num << std::endl;
  by_ref(ref_num);
  by_ptr(&ptr_num);
  std::cout << ref_num << std::endl;
  std::cout << ptr_num << std::endl;
}
```

```
5
5
3
3
```

<br>

### `cin`과 `scanf`

C에서는 어떤 변수의 값을 다른 함수에서 변경시키기 위해 포인터를 사용했으며, 이는 `scanf` 함수에서도 알 수 있다.

```c
scanf("%d", &user_input);
```

C++에서 사용하는 `cin` 또한 주소값을 필요로하는 것은 동일하다. 하지만, `cin`의 경우 파라미터를 reference로 받도록 되어 있기 때문에, 다음과 같이 작성하는 것 만으로도 충분하다.

```cpp
std::cin >> user_input;
```

<br>

<br>

## Reference 대상

### Literal 참조

만약 아래와 같이 참조를 시도한다면, 리터럴이 수정될 여지가 있기 때문에 에러가 발생한다.

```cpp
int &ref = 4;
```

하지만, `const`를 사용한다면 리터럴도 참조할 수 있다.

```cpp
#include <iostream>

int main() {
    const int& ref = 4;
    int a = ref;
    std::cout << a << std::endl;
}
```

```
4
```

위 코드는 `int a = 4`를 사용하는 것과 동일한 역할을 하게 된다.

<br>

### 배열 참조

아래 코드 블록과 같이 레퍼런스의 배열을 생성하려고 하면 에러가 발생한다.

```cpp
int a, b;
int& arr[2] = {a, b};
```

왜냐하면, 레퍼런스는 주소값을 갖지 않기 때문이다. 위 코드는 `a`와 `b`를 묶어 배열로 만들고, 이를 `arr`라는 배열에 저장하려고 한다. 하지만 레퍼런스는 주소값을 가질 수 없으므로, 새로운 구조체를 저장할 수 없다.

 따라서 `a`와 `b`를 원소로 갖는 배열은 앞에서 먼저 선언한 후, 해당 배열을 참조하는 레퍼런스를 생성해야 한다. 배열을 참조하기 위해서는, 참조하는 **배열의 크기를 명시**해주어야 한다.

```cpp
#include <iostream>

int main() {
  int arr[3] = {1, 2, 3};
  int(&ref)[3] = arr;

  ref[0] = 2;
  ref[1] = 3;
  ref[2] = 1;

  std::cout << arr[0] << arr[1] << arr[2] << std::endl;
  return 0;
}
```

```
231
```

만약 배열이 2차원 형태를 가질 경우에도 아래와 같이 명시해 주어야 한다.

```cpp
int arr[3][2] = {1, 2, 3, 4, 5, 6};
int (&ref)[3][2] = arr;
```

따라서 배열을 가리켜야 할 때에는 `int *p` 하나로 모든 1차원 배열들을 가리킬 수 있는 포인터를 사용할 것을 권장한다.

<br>

<br>

## Reference 반환

아래의 코드 블록에서, 만약 `fn1(x)`를 사용할 경우 `x`의 값이 반환될 뿐 `x` 자체가 반환되지 않는다. 즉, literal 값인 1이 반환되므로, `++`을 사용하려 하면 에러가 발생한다. 하지만 `fn2(x)`를 사용할 경우, `x`의 주소값을 반환하므로 `x` 변수에 `++`이 적용된다.

```cpp
#include <iostream>

int fn1(int &a) { return a; }
int &fn2(int &a) { return a; }

int main() {
	int x = 1;
	std::cout << fn1(x)++ << std::endl; // 에러 발생
	std::cout << fn2(x)++ << std::endl;
	std::cout << x << std::endl;
}
```

```bash
# 에러가 발생하는 fn1(x)++를 제외시키고 컴파일한 내용
1
2
```

<br>

### Reference가 메모리에 존재하는가?

Reference는 함수 호출로 인해 호출 스택이 달라질 때, 해당 메모리에 접근하기 위해 주소가 필요하다. 따라서 이러한 경우 주소 메모리 공간이 필요하다. 만약 호출 스택이 같다면, 바로 접근할 수 있으며, 별도의 주소 메모리 공간이 필요 없다.

1. 메모리에 존재하는 경우란, 즉 참조가 유효한 경우, 같은 함수 내에 있는 경우 혹은 runtime이 해당 함수를 실행하는 때 일 경우라는 뜻이죠. 간단한 예로 main 에 선언된 참조는 선언 순간부터 main함수가 끝날 때 까지 대상 변수와 같은 저장 공간에 공존해 있습니다.
2. 여기서! 메모리에 존재하지 않는 경우란, 즉 해당 함수가 끝나고 runtime 이 다음 영역으로 갔을때를 말합니다. 위의 예제대로 `int& a`는 `fn1`이건 `fn2` 이건, 호출시 `x`의 저장공간에 세내고 묵고있다가 함수값 반환과 동시에 방빼고 쫓겨납니다. 여기서 차이는 반환값이죠. `fn1`의 반환값은 참조가 아니기 때문에 해당 소스에서는 함수값 반환과 동시에 `x`는 방혼자쓰는 쏠로가 됩니다. 하지만 `fn2`의 경우 반환값이 참조형식이기 때문에 `a`는 사라졌을지언정, `fn2`가 `x`의 참조로서 남아있습니다. 정확히는 `fn2(x)`인데, 이 동거인은 `main`함수에 반환되었기 때문에 `main`이 끝날때까지 살아있습니다. 간단한 실험으로 한번 `fn2(x)`를 `for`문에 `++`로 다섯번정도 증감시켜 보세요. 그 후, `x`나 `fn2(x)`를 출력하면 값에 5가 증가한 것을 보실 수 있습니다. 거기에 또! `fn2`에 `y`라는 새로운 변수를 넣어서 `fn2(y)`라는 참조연산자를 새로만들고, `fn2(x)`랑 섞어서 이것저것 증감해보세요. 컴파일해 보시면 아시겠지만, `y`는 `y`대로, `x`는 `x`대로 값이 증감되있는 것을 확인하실 수 있으실 겁니다.

<br>

<br>