// 
#define _CRT_SECURE_NO_WARNINGS
// in/out 기능을 사용하기 위해
#include <iostream>
// std namespace를 사용함을 선언하지 않는다면, 이후 추가적인 코드 부여 필요
using namespace std;

//// global variable declaration
// int T, ans, N, M;

//// asterisk는 포인터 변수 앞에 붙여주는데,
//// 이는 asterisk의 개수 만큼 안쪽에 실제 데이터가 존재함을 의미한다.
//// C++에서 2차원 배열을 구현할 때, 1차원 배열의 데이터는 모두 포인터 변수들이며
//// 2차원 배열의 내부 데이터들이 실제 데이터이다.
//// 따라서 2차원 배열을 선언할 때에는 ** 를 붙여주어야 한다(*위치에는 데이터가 아닌 포인터 변수들이 존재)
int **B;
bool *visited;

class Node {
public:
    // Instance Variable 선언
    // 인스턴스 변수와 초기화 함수를 private에 포함시킨다면, 객체 생성 및 조작 불가
    int data;
    Node *pre;
    Node *nxt;

    //// Node class의 초기화 함수
    Node(int data) {
        this->data = data;
        this->pre = NULL;
        this->nxt = NULL;
    }
};

//// Doubly Linked List의 head와 tail을 관리
// 실질적으로 DLL을 보유하고 있는 클래스가 아니며,
// 단순히 중간 관리자 역할을 수행
class DoublyLinkedList {
private:
    int size;
    Node *head;
    Node *tail;
public:
    // stack class의 초기화 함수
    DoublyLinkedList() {
        this->size = 0;
        this->head = NULL;
        this->tail = NULL;
    }

    void push(int data) {
        //// node라는 이름으로 새 Node 객체를 생성하며, head와 tail을 변경하는 과정
        // System Memory Stack에 저장되는 *element 포인터 변수는
        // Heap에 저장되는 동적 할당된 객체의 주소를 가리킨다.
        // Node(data)를 수행하여 Node 클래스의 초기화 함수를 data 값을 부여하여 실행한다.
        Node *node = new Node(data);
        
        // 첫 노드 생성
        if (size == 0) {
            // 첫 노드를 DLL 전체의 head와 tail에 할당
            this->head = node;
            this->tail = node;
        }
        else {
            // 1. 새 노드의 이전 노드로, 현재의 tail 노드를 지정
            node->pre = this->tail;
            // 2. 현재의 tail 노드의 다음 노드로 새 노드를 지정
            this->tail->nxt = node;
            // 3. tail을 새 노드로 지정
            this->tail = node;
        }
        this->size++;
    }

    int Size() {
        return this->size;
    }

    int Pop() {
        if (this->size == 0)
            return -1;

        int popped = this->tail->data;
        if (this->size == 1) {
            delete this->tail;
            this->head = NULL;
            this->tail = NULL;
            
        }
        else {
            // tail을 tail의 prev로 변경
            this->tail = this->tail->pre;
            // 변경된 현재의 tail의 next(변경 이전의 tail)의 동적 할당된 데이터를 제거
            delete this->tail->nxt;
        }
        this->size--;
        return popped;
    }
};

int main() {
    //// C++은 다른 언어와 달리 I/O 장비를 이용할 때 객체를 전달하기 때문에 속도가 느리다.
    //// 알고리즘 테스트를 위해서는 속도를 향상시키기 위해 다음 두 줄의 코드를 사용하여 이를 해결 가능
    // ios::sync_with_stdio(false);
    // cin.tie();

    //// file을 open하는 것이며, 앞에서 using namespace std;를 선언하였기 때문에 이처럼 간결하게 할 수 있다.
    //// freopen()은 C에서 또한 사용하지만, 보안 상의 이유로 C++ freopen_s()를 사용한다.
    //// 하지만 이를 위해 추가적으로 기입해야 하는 코드가 길어지기 때문에 미리 using namespace std;를 작성하여 간소화한다.
    // freopen("input.txt", "r", stdin);
    // cin >> T;
    // for (int t=1;t<=T;t++) {}



}