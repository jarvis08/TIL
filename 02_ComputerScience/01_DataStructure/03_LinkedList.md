# 연결 리스트, Linked List

---

> 객체가 선형적 순서를 가지도록 배치된 자료구조
>
> 인덱스에 의해 선형적 순서가 결정되는 배열과 달리, 각 객체에 있는 포인터에 의해 순서를 결정
>
> 동적 집합을 위한 단순하고 유연한 표현 방법을 제공하며, 다양한 연산을 지원 가능
>
> But, 가능하다 != 효율적이다

## Linked List의 종류

- Linked List의 형태 결정 요소

  1. 단순 연결 리스트, Singly Linked List

     각 원소의 `prev` 포인터가 제외된 리스트

  2. 양방향 연결 리스트, Doubly Linked List

  3. 정렬(Sorted) 여부

     - 정렬(Sorted) 연결 리스트

       리스트의 순서가 각 원소의 키 순서대로 저장되어 있는 연결 리스트

       최소값을 가지는 원소가 리스트의 `head`에 위치하며, 최대값의 원소가 `tail`에 위치

     - 비정렬 연결 리스트

       원소가 아무 위치에나 존재

       e.g., 비정렬 양방향 리스트

  4. 환형 연결 리스트, Circular Linked List

     리스트 `head`의 `prev` 포인터가 `tail`을 가리키고, 리스트 `tail`의 `next` 포인터가` head`를 가리킴

---

## 양방향 연결 리스트, Doubly Linked List

- Doubly Linked List

  `L`(Doubly Linked List)의 각 원소는 `key` 속성값과 두 개의 포인터인 `prev`와 `next`를 속성값으로 가지는 객체

  - 객체는 부가 데이터를 가질 수 있다.

  - 원소 `x`가 있을 때,

    `x.next` : linked list의 바로 다음 원소를 지칭

    `x.prev` : 바로 직전의 원소를 지칭

- `x.prev = NIL` 인 경우 원소 `x`는 바로 이전 원소가 없으므로,

  이를 linked list의 **첫 번째 원소** 또는 **머리(head)** 라고 부른다.

- `x.next = NIL` 경우 원소 `x`는 다음 원소가 없으므로,

  이를 linked listd의 **마지막 원소** 또는 **꼬리(tail)** 라고 부른다.

- `L.head`는 리스트의 **첫 번째 원소**를 가리키며,

  `L.head = NIL`인 경우는 **리스트가 비었음**을 의미

---

## Linked List에서의 Search

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

## Linked List로의 Insert

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

## Linked List에서의 Delete

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

## 경계 원소

