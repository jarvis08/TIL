# 단순 연결 리스트, Singly Linked List

## 정의 및 특징

- 구조

  - 노드가 하나의 Link Field에 의해 다음 노드와 연결되는 구조

    C의 경우 `자기 참조 구조체` 사용이 필요

  - 헤드가 처음 노드를 가리키고, Link Field가 연속적으로 다음 노드를 가리킴

  - 최종적으로 NULL을 가리키는 노드가 가장 마지막 노드

- **새 node인 new를 linked list에 삽입하는 과정**

  1. `new`의 `link field`에 기존 노드의 `data field`(=`이전 순서 노드의 link field`)를 할당
  2. `new`의 이전 순서 노드의 `link field`에 `new의 주소`를 할당
  
  ```python
  # Head 뒤에 node(new)를 삽입하는 과정
  new.data = 'AA'
  new.link = head.link
  head.link = new
  ```
  
  - **link 값의 할당**을 가장 먼저 생각, 처리하는게 중요
  
- 데이터 참조하기

  ```python
  # lined_list[3] 구하기
  value = head.link.link.link
  ```

  Linked List가 깊어지면 `.link`를 무수히 많이 붙여야 하기 때문에 이를 `반복문`으로 처리

  ```python
  # 반복문으로 SLL[3] 구하기
  pre = Head
  for i in range(3):
      pre = pre.link
  ```

  ```python
  # 마지막 노드 찾기
  pre = Head
  while pre.link != None:
      pre = pre.link
  ```



## 구현 함수

### Insert

- 첫 번째 노드로 삽입

  ```python
  def addtoFirst(data):
      global Head
      Head = Node(data, Head)
  ```

- 가운데 노드로 삽입

  `pre`라는 노드의 다음 위치에 새로운 노드를 삽입

  ```python
  def add(pre, data):
      if pre == None:
          print('error')
      else:
          pre.link = Node(data, pre.link)
  ```

- 마지막 노드로 삽입

  ```python
  def addtoLast(data):
      global Head
      if Head == None:
          Head = Node(data, None)
      else:
          p = Head
          # 계속 .link를 할당하여 마지막 노드를 탐색
          while p.link != None:
              p = p.link
          # 새로운 노드 할당
          p.link = Node(data, None)
  ```



### Delete

- 노드 삭제

  ```python
  # pre의 다음 노드를 삭제하는 함수
  def delete(pre):
      if pre == None or pre.link == None:
          print('error')
      else:
          pre.link = pre.link.link
  ```

  Python의 경우 사용되지 않는 노드를 Garbage Collector가 알아서 정리

  C/C++의 경우 개발자가 직접 Free
