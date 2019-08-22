# Stack

- 물건을 쌓아 올리듯 자료를 쌓아 올린 형태의 구조

- 선형 구조

  자료 간의 관계가 1대1

- 후입 선출(LIFO, Last-in Frist-out)

  마지막으로 삽입한 자료를 가장 먼저 꺼냄

- 저장소 자체를 stack이라 부르기도 함

## 구현

- `top` : 마지막으로 삽입된 원소의 위치

- 연산

  - 삽입, `push`
  - 삭제, `pop`
  - `isEmpty` : Underflow 여부 확인
  - `isFull` : Overflow 여부 확인
  - `peek` : `top`에 있는 item을 반환

- 구현

  static하게 memory에 직접 접근해야 빠르므로, `append` 사용은 자제하는 것이 좋다.

  ```python
  # overflow와 underflow를 고려하지 않은 코드
  stack = [0] * 10
  top = -1
  
  # push
  for i in range(3):
  	stack[top+1] = i
  	top += 1
  
  # pop
  # pop 메쏘드를 사용할 수도 있다.
  for i in range(3):
      t = stack[top]
      top -= 1
      print(t)
  ```

- 구현 주의 사항

  - 1차원 배열을 사용하여 구현할 경우 구현이 용이하다는 장점이 존재하지만,

    스택의 크기를 변경하기가 어렵다는 단점

  - 연결 리스트를 사용하여 메모리를 관리하는 것이 효율적

## 응용

- 괄호 검사

  `(`와 `)`는 언제나 pair로 동작

  `(`를 stack에 `push`하던 중, `)`가 나올 때 마다 `pop`

- Function Call

  프로그램에서의 함수 호출과 복귀에 따른 수행 순서를 관리

  (함수 호출 순서의 역순으로 `return`을 수행)

  - 컴퓨터의 메모리

    - Code

    - Data

      전역 변수

    - Heap & Stack

      local 변수, 함수 호출 시 함수의 내용