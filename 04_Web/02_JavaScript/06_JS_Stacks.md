# JavaScript Stacks

참고 자료: [Kuhnhee](https://github.com/Kuhnhee/TIL/tree/master/School/Javascript)

JavaScript에는 세 가지 메모리 영역이 있습니다.

1. Call Stack
2. Event(Task) Stack
3. Heap

<br>

## Call Stack

Call Stack에서는 scipt 내에 생성되어 있는 함수들이 기록됩니다.

<br>

<br>

## Event(Task) Stack

Scipt 내에 비동기(Asynchronous) 함수가 있을 경우, 우선 event stack 혹은 task stack이라고 불리는 영역에 저장합니다. 이후 Call Stack에 저장된 함수들이 모두 종료되면 event stack의 함수들을 하나씩, 차례로 call stack으로 가져와서 실행하게 됩니다.

<br>

<br>

### Heap

Script의 내용을 수행하며 동적으로 생성된 객체(인스턴스)들을 heap에 할당합니다. JavaScript 뿐만 아니라, 대부분의 *구조화되지 않은 '더미'같은 메모리 영역*을 heap이라 표현합니다.