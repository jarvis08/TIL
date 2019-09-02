# 이중(양방향) 연결 리스트, Doubly Linked List

## Doubly Linked List의 기본 요소

`L`(Doubly Linked List)의 각 원소는 `key` 속성값과 두 개의 포인터인 `prev`와 `next`를 속성값으로 가지는 객체

- `prev`

  이전 노드의 `prev` 주소를 가리키는 Link Field

- `data`

- `next`

  다음 노드의 주소를 가리키는 Link Field

- 객체는 부가 데이터를 가질 수 있다.

- 원소 `x`가 있을 때,

  `x.next` : linked list의 바로 다음 원소를 지칭

  `x.prev` : 바로 직전의 원소를 지칭
  
- `NIL`

  - `x.prev = NIL` 인 경우

    원소 `x`는 바로 이전 원소가 없으므로, 이를 linked list의 **첫 번째 원소** 또는 **머리(head)** 라고 부른다.

  - `x.next = NIL` 경우

    원소 `x`는 다음 원소가 없으므로, 이를 linked listd의 **마지막 원소** 또는 **꼬리(tail)** 라고 부른다.

  - `L.head`는 리스트의 **첫 번째 원소**를 가리키며,

    `L.head = NIL`인 경우는 **리스트가 비었음**을 의미



## Doubly Linked List의 사용

- Head는 첫 번째 원소의 주소(Linke Filed만이 존재)만을 보유

- 첫 번째 원소의 `prev`는 `null`

- `cur` 뒤에 `new` 원소 삽입 과정

  1. `new`를 생성하고 data field에 값을 저장

  2. `cur`의 `next`를 `new`의 `next`에 할당

     원래 오른쪽에 위치했던 노드는 new에게도 오른쪽에 있다고 인식시킴

  3. `new`의 `prev`주소를 `cur`의 `next`에 할당

     cur에게 new가 오른쪽 노드라고 인식시킴

  4. `new`의 `prev`에 `cur`의 `prev` 주소를 할당

     new에게 왼쪽 노드는 cur임을 인식시킴

  5. `new`의 다음 노드의 `prev`에 `new`의 `prev` 주소를 할당

     오른쪽 노드에게 왼쪽에 new가 있음을 인식시킴

- 원소 삭제 과정

  삭제할 노드(`cur`)의 좌우 노드들을 서로 연결시켜줌

  1. `cur`의 오른쪽 노드의 주소를 `cur`의 왼쪽 노드의 `next`에 저장
  2. `cur`의 왼쪽 노드의 주소를 `cur`의 오른쪽 노드의 `pre`에 저장
  3. `cur`의 메모리 반환



### 함수 - Search

- `LIST-SEARCH(L, k)` procedure는 단순 선형 검색을 통해 리스트 `L`에서 키 `k`를 가지는 첫 번째 원소를 찾아 그 포인터를 리턴

- 키 `k`를 갖는 원소가 존재하지 않으면 `NIL` 값을 리턴

- `LIST-SEARCH(L, k)`

  ```
  x = L.head
  while x != NIL and x.key != k
      x = x.next
  return x
  ```

  - n개의 객체를 가지는 리스트를 검색할 때 LIST-SEARCH procedure는 최악의 경우 리스트의 모든 원소를 검색

    수행 시간 - **O(n)**



### 함수 - Insert

- `LIST-INSEART`는 `key` 속성이 미리 채워진 원소 `x`를 linked list의 맨 앞에 이어붙인다.

  속성값 표현은 연속해서 뒤로 이어질 수 있으므로, `L.head.prev`는 `L.head`가 가리키는 객체의 속성값 `prev`를 의미

  - `LIST-INSERT(L, x)`

    ```
    x.next = L.head
    if L.head != NIL
        L.head.prev = x
    L.head = x
    x.prev = NIL
    ```

    - n개의 원소를 가지는 리스트에 대해 LIST-INSERT procedure의 수행시간 - **O(1)**



### 함수 - Delete

- `LIST-DELETE` procedure는 linked list `L`에서 원소 `x`를 삭제

  삭제를 하기 위해 원소 `x`의 포인터가 필요하며, 원소의 포인터를 알기 위해 `LIST-SEARCH`를 호출

  - `LIST-DELETE(L, x)`

    ```
    if x.prev != NIL
        x.prev.next = x.next
    else L.head = x.next
    if x.next != NIL
        x.next.prev = x.prev
    ```

    삭제 자체의 수행 시간 - **O(1)**

    `LIST-SEARCH`를 호출을 포함한 수행 시간 - **O(n)**
