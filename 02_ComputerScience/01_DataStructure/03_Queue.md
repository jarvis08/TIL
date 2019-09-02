# 큐, Queue

### Queue 특성

- 큐의 FIFO 특성은 계산대에 줄을 서  있는 사람을 서비스하는 일과 유사

  - INSERT - ENQUEUE
  - DELETE - DEQUEUE
- 머리(`head`)와 꼬리(`tail`) 인자

  - 새로 도착한 손님이 줄의 맨 끝에 위치하는 것 처럼 새 원소는 꼬리(`tail`)에 위치
  - 큐의 삭제는 가장 오래 기다린, 대기열 맨 앞의 머리(`head`)
  - 머리를 `front`, 꼬리를 `rear`로 지칭하기도 한다.
  - `front`/`head`를 `첫 원소 index-1`, 그리고 `tail`/`rear`를 `마지막 원소의 index`로 처리하기도 한다.



### Queue 구현

- `Q[1, 2, …, n]` 원소가 최대 n-1개

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



### Queue 연산

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