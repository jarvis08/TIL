# Applications of Stack

## Calculator

실제 System Stack에서는 계산식이 들어오면 **후위 표기식으로 변환하여 계산**

- `4 + 3 * 5`

  - 토큰 5개

    - Token Analyzing

      프로그래밍의 Token :: 의미가 있는 가장 작은 단위

  - 피연산자 3개

  - 연산자 2개

    - 단항 연산자 0개
    - 이항 연산자 2개
    - 중위 표기법

- 식의 표기법

  - 전위 표기법

  - 중위 표기법, Infix Notation

    연산자를 피연산자의 가운데 표기하는 방법

  - 후위 표기법, Postfix Notation

    연산자를 피연산자 뒤에 표기하는 방법

- 중위 표기식을 후위 표기식으로 변환하기

  1. 수식의 각 연산자에 대해 우선순위에 따라 괄호를 사용하여 다시 표현
  2. 각 연산자를 그에 대응하는 오른쪽괄호의 뒤로 이동
  3. 괄호 제거

  ```
  e.g., A*B-C/D
  1. ((A*B)-(C/D))
  2. ((A B)* (C D)/)-
  3. AB*CD/-
  ```

- Stack 사용하여 **중위 표기식을 후위 표기식으로** 변환하기

  - Stack에 계속해서 **연산자와 괄호**를 `push()`, 피연산자는 바로 출력
  - 괄호가 끝나는 시점마다 괄호 사이의 연산자를 `Stack.top`부터 출력
  - 현재 순서의 연산자가 `Stack.top` 보다 우선순위가 높을 시 `Stack.push()`
  - 현재 순서의 연산자가 `Stack.top` 보다 우선순위가 같거나 낮으면 `Stack.top`을 출력 후 `push()`

  ```
  (6 + 5 * (2 - 8) / 2)
  >> 6528-*2/+
  ```

- Stack을 사용하여 **후위 표기식 계산**하기

  1. 후위 표기식으로의 변환 알고리즘과 반대로 Stack에 **피연산자**를 `push()`
  
  2. 현재 순서가 연산자일 경우 `Stack.top`을 2회 `pop()`하여 연산
  
     `Stack.top-1 (연산자) St ack.top`의 **계산 결과**를 `push()`