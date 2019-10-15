# 연결 리스트, Linked List

## Linked List

### 정의

- 객체가 선형적 순서를 가지도록 배치된 자료구조
- 인덱스에 의해 선형적 순서가 결정되는 배열과 달리, 각 객체에 있는 포인터에 의해 순서를 결정
- 동적 집합을 위한 단순하고 유연한 표현 방법을 제공하며, 다양한 연산을 지원 가능
- But, `가능하다 != 효율적이다`

<br>

### 종류

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

<br>

<br>

## Linked List의 활용

### 삽입 정렬

도서관 사서가 책을 정렬하는 방법

자료 배열의 모든 원소들을 **앞에서부터** 차례대로 **이미 정렬된 부분과 비교**하여 자신의 위치를 찾아나감

**자료의 삽입과 삭제가 빈번하게 발생**하므로, 일반 리스트 사용 시 불필요한 인덱싱 작업이 무수히 반복되므로 **연결 리스트**를 활용

<br>

### 병합 정렬

자료를 최소 단위의 문제까지 나눈 후, 차례대로 정렬하여 최종 결과를 도출 (Top-Down 방식)

시간 복잡도 : `O(nlogn)`

- 기존의 병합 방법

  1. 분할된 리스트 두 곳에 모두 Queue의 front의 역할을 하는 i, j를 부여

  2. i와 j 인덱스의 값을 각각 비교하며 작은 값을 병합 list에 할당
  3. 작은 값에 해당하는 index + 1

- Linked List 사용

  4. 만약 i,j 중 한 리스트의 처리가  끝난다면, 나머지 값들은 모두 뒤에 이어 붙인다.

     이 때 `병합[idx] =  list[i]` 를 반복하여 넣는 것이 아니라,

     `list[i]`의 주소값을 이용하여 그 뒷부분까지 모두 한번에 할당

<br>

### 스택, Stack

스택 내의 **순서**를 Linked List로 구현

- 구성 요소

  - `Push()`

    List의 마지막에 노드를 삽입

  - `Pop()`

    리스트의 마지막 노드를 반환 및 삭제

  - `top`

    List의 마지막(최근에 삽입된) 노드를 가리키는 변수

  - `top == null`

    초기 상태

- 구현

  1. null 값을 가지는 노드를 만들어 스택 초기화
  2. 원소 삽입
     1. top의 주소를 new의 주소로 할당
     2. `new의 link field == null`
  3. 원소 삽입
     1. top의 주소를 new의 주소로 할당
     2. `new의 link field == 2번에서 삽입한 노드 주소`

  ```python
  def push(i):
      global top
      top = Node(i, top)
      
      
  def pop():
      global top
      if top == None:
          print('error')
      else:
          data = top.data
          top = top.link
          return data
  ```

<br>

### 우선 순위 큐, Priority Queue

Queue 내부를 원하는 형태로 Sorting하며, 그 과정은 **삽입 정렬**과 동일

**Heap Tree**를 활용하여 구현하는 것이 Linked List보다 효율적

