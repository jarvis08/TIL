# 1-6. 하드웨어 보호, HW Protection

운영체제는 이중 모드를 활용하여 하드웨어를 보호한다.

<br><br>

## 1-6-1. 입/출력 장치 보호, Input/Output Device Protection

사용자의 잘못된 입출력 명령을 방지

- 다른 사용자의 입출력, 정보 등이 침해하려는 경우
- 예시
  - printer 혼선
  - reset 명령
  - 다른 사용자의 파일 읽고 쓰기

<br>

### 보호 방법

OS는 입출력을 특권 명령(Previleged Instruction)으로만 사용 가능하게 하여 위의 문제들을 방지한다.

- 특권 명령(Previleged Instruction)의 예시
  - `ldr`: 메모리의 data를 register로 가져옴
  - `stop`: CPU 중지
  - `reset`: 전체 시스템 초기화
  - `in`: 키보드 등의 입력 장치의 명령을 가져옴
  - `out`: 프린터, 모니터와 같은 출력 장치를 조작

사용자(, 사용자 프로그램)가 입출력 명령을 직접 내릴 경우 **Privileged Instruction Violation** 발생

<br>

<br>

## 1-6-2. 메모리 보호, Memory Protection

어느 사용자 프로그램이 다른 사용자의 메모리, 혹은 운영체제 메모리 영역으로 접근 시 차단한다.

i.g., 해킹 프로그램이 운영체제 메모리 영역의 ISR을 조작하려 시도

i.g., 다른 프로그램이 어떤 프로그램의 메모리에 접근

<br>

### 보호 방법

CPU는 Main Memory에 **Address Bus**를 통해 특정 주소의 데이터를 달라고 요청하며, 메인 메모리는 **Data Bus**를 통해 요청 받은 주소의 데이터를 반환한다. 운영체제는 Address Bus를 전송하는 시점에 MMU(Memory Management Unit)을 반드시 거치게 하여, 허용되지 않은 주소의 데이터를 요청(**Segment Violation**)하면 CPU에 Interrupt를 발생시킨다.

<br>

<br>

## 1-6-3. CPU 보호, CPU Protection

어느 사용자 혹은 프로그램이 CPU를 독점하여 사용하려 하는 경우를 방지한다.

프로그래밍 언어의 `while True:`와 같은 표현은 CPU 독점을 야기할 수 있다. 따라서 **Timer Interrupt**를 적용하여 일정 시간 경과 시 강제로 다른 프로그램으로 전환한다.

