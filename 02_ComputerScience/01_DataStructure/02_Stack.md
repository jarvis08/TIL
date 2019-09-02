# 스택, Stack

### Stack 특성

- Stack의 연산, 식당에서 스프링으로 접시를 받쳐 올려주는 접시 스택에서 유래
  - INSERT - PUSH
  - DELETE - POP

- 선형 구조

  자료 간의 관계가 1대1

- 후입 선출(LIFO, Last-in Frist-out)

  마지막으로 삽입한 자료를 가장 먼저 꺼냄

- 저장소 자체를 stack이라 부르기도 함



### Stack 구현

- `S[1, 2, …, n]`

  최대 n개의 원소를 가지는 스택 집합

- `S.top` 속성

  가장 최근에 삽입된 원소를 지칭, S[1, 2, …, S.top]

- `S.top == 0`

  원소를 갖고있지 않은 빈(empty) 상태

  - `STACK_EMPTY` 연산을 통해 검사

  - 스택 부족, underflow

    빈 스택에서 원소추출 시

  - 스택 포화, overflow

    `S.top > n` 일 시

- Stack의 연산, O(1)

  - STACK-EMPTY(S)

    ```
    if S.top == 0
    	return TRUE
    else return False
    ```

  - PUSH(S, x)

    ```
    S.top = S.top + 1
    S[S.top] = x
    ```

  - POP(S, x)

    ```
    if STACK-EMPTY(S)
    	error "스택 부족"
    else S.top = S.top - 1
    	return S[S.top+1]
    ```
  
  - `isEmpty` : Underflow 여부 확인
  - `isFull` : Overflow 여부 확인
  - `peek` : `top`에 있는 item을 반환

- 구현 주의 사항

  - 1차원 배열을 사용하여 구현할 경우 구현이 용이하지만,

    스택의 크기를 변경하기가 어렵다는 것이 단점

  - 연결 리스트를 사용하여 메모리를 관리하는 것이 효율적



### Stack 응용

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
