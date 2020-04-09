# Friend

출처: [열코의 프로그래밍 일기](https://yeolco.tistory.com/116)

## Friend Class

`firend` 클래스는 `friend`로 선언된 다른 클래스의 private 및 protected 멤버에 접근할 수 있다. 

```cpp
#include <iostream>
#include <string>
using namespace std;
 
class Friend1 {
private :
    string name;
 
  	// Friend2 클래스를 friend로 지정
    friend class Friend2;
};
 
class Friend2{
public : 
    void set_name(Friend1& f, string s) {
        f.name = s;
    }
    void show_name(Friend1& f) {
        cout << f.name << "\n";
    }
};
 
int main(void) {
    Friend1 f1;
    Friend2 f2;
 		
  	// Friend2 클래스의 인스턴스가 Friend1의 멤버 함수를 사용
    f2.set_name(f1, "열코");
    f2.show_name(f1);
 
    return 0;
}
```

<br>

### Friend Function

friend 함수를 사용하면, 특정 함수만을 사용할 수 있도록 제한적으로 허가할 수 있다.

```cpp
#include <iostream>
#include <string>
using namespace std;
 
class Friend1 {
private :
    string name;
 		
  	// set_name 함수만을 friend에게 허용
    friend void set_name(Friend1&, string);
};
 
 
void set_name(Friend1& f, string s) {
    f.name = s;
    cout << f.name << "\n";
}
 
int main(void) {
    Friend1 f1;
 
    set_name(f1, "열코");
 
    return 0;
}
```

주의사항은 다음과 같다.

- 친구의 친구는 friend 키워드로 명시하지 않은 경우 친구 관계가 형성되지 않는다.
- 친구의 자식도 마찬가지로 friend 키워드로 명시하지 않은 경우 친구 관계가 형성되지 않는다.