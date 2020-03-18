# Modern Range-based `for` Loop

```cpp
#include <iostream>
#include <vector>
int main() {
  
  // In the standard library, a std::vector is an array with automatic size.
  // Let's make a vector of ints and loop over the contents.
  // The syntax for std::vector<> is discussed further in the lecture on template types.
  
  std::vector<int> int_list;
  int_list.push_back(1);
  int_list.push_back(2);
  int_list.push_back(3);
  
  // Automatically loop over each item, one at a time:
  for (int x : int_list) {
    // This version of the loop makes a temporary copy of each
    // list item by value. Since x is a temporary copy,
    // any changes to x do not modify the actual container.
    x = 99;
  }
  
  for (int x : int_list) {
    std::cout << "This item has value: " << x << std::endl;
  }
  
  std::cout << "If that worked correctly, you never saw 99!" << std::endl;

  return 0;
}
```

```
This item has value: 1
This item has value: 2
This item has value: 3
If that worked correctly, you never saw 99!
```

<br>

## Reference 활용

### 배열에 값 저장하기

배열의 요소들에 `99`라는 값을 저장시켜 보겠습니다.

```cpp
#include <iostream>
#include <vector>
int main() {
  
  std::vector<int> int_list;
  int_list.push_back(1);
  int_list.push_back(2);
  int_list.push_back(3);
  
  for (int& x : int_list) {
    // This version of the loop will modify each item directly, by reference!
    x = 99;
  }
  
  for (int x : int_list) {
    std::cout << "This item has value: " << x << std::endl;
  }
  
  std::cout << "Everything was replaced with 99!" << std::endl;

  return 0;
}
```

```
This item has value: 99
This item has value: 99
This item has value: 99
Everything was replaced with 99!
```

<br>

### 큰 리스트 사용하기

만약 `inst_list`의 크기가 너무 방대할 경우, reference를 이용할 수도 있습니다.

```cpp
#include <iostream>
#include <vector>
int main() {
  
  std::vector<int> int_list;
  int_list.push_back(1);
  int_list.push_back(2);
  int_list.push_back(3);
  
  for (const int& x : int_list) {
    // This version uses references, so it doesn't make any temporary copies.
    // However, they are read-only, because they are marked const!
    std::cout << "This item has value: " << x << std::endl;
    // This line would cause an error:
    //x = 99;
  }

  return 0;
}
```

```
This item has value: 1
This item has value: 2
This item has value: 3
```



