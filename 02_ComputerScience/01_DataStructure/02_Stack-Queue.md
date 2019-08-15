# 스택과 큐, Stack and Queue

---

## 스택, Stack

- Stack의 연산, 식당에서 스프링으로 접시를 받쳐 올려주는 접시 스택에서 유래
  - INSERT - PUSH
  - DELETE - POP

- S[1, 2, …, n]

  최대 n개의 원소를 가지는 스택 집합

- S.top 속성

  가장 최근에 삽입된 원소를 지칭, S[1, 2, …, S.top]

- S.top == 0

  원소를 갖고있지 않은 빈(empty) 상태

  - STACK_EMPTY 연산을 통해 검사

  - 스택 부족, underflow

    빈 스택에서 원소추출 시

  - 스택 포화, overflow

    S.top > n 일 시

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

---

## 큐, Queue

- 큐의 FIFO 특성은 계산대에 줄을 서  있는 사람을 서비스하는 일과 유사

  - INSERT - ENQUEUE
  - DELETE - DEQUEUE

- 머리(head)와 꼬리(tail) 인자

  - 새로 도착한 손님이 줄의 맨 끝에 위치하는 것 처럼 새 원소는 꼬리(tail)에 위치
  - 큐의 삭제는 가장 오래 기다린, 대기열 맨 앞의 머리(head)

- 구현 방법

  - Q[1, 2, …, n] 원소가 최대 n-1개

    `Q.tail`이 새로 저장될 원소의 공간을 지시하므로, 최대 n개의 원소가 아닌 n-1

  - `Q.head`

    머리를 가리키는 인덱스, 포인터의 속성값

  - `Q.tail`

    새로운 원소가 삽입될 위치의 속성값

  - 큐의 원소들은 `Q.head`,` Q.head_1`, …, `Q.tail-1` 위치에 존재

    고리모양의 순환 순서로, tail이 Q[n]의 위치 이후에는 다시 Q[1]로 이동

  - `Q.head == Q.tail` 일 경우 큐는 비어있다.

    초기에는 `Q.head = 1`,` Q.tail = 1`로 시작

  - 큐 부족(underflow)

    빈 큐에서 원소 삭제 시도

  - 큐 포화(overflow)

    `Q.head = Q.tail + 1`

    `Q.head = 1` `and` `Q.tail = Q.length`

    위 두 경우는 큐가 가득찬 상태이며, 추가적인 원소 삽입 시 큐 포화(overflow)

- 큐의 연산

  - ENQUEUE(Q, x)

    ```
    Q[Q.tail] = x
    if Q.tail == Q.length
    	Q.tail = 1
    else Q.tail = Q.tail + 1
    ```

  - DEQUEUE(Q)

    ```
    x = Q[Q.head]
    if Q.head == Q.length
    	Q.head = 1
    else Q.head = Q.head + 1
    return x
    ```
