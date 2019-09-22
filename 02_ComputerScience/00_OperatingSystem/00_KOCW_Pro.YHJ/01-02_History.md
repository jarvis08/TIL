# 1-2. 운영체제의 역사

### No Operating System

- 운영체제의 형태

  컴퓨터가 탄생한 1940년대 말 즈음에는 운영체제가 존재하지 않았다.

- 컴퓨터의 구조

  - Card Reader

    - 연구자가 작성한 코드에 해당하는 카드
    - 코드 실행을 위한 필수 기능을 실행하는 카드
      - Compile
      - Link
      - Load

  - Memory

  - Processing

  - Line Printer

    망치로 종이를 두들겨 찍어냄

<br>

### 일괄처리시스템, Batch Processing System

- 운영체제의 형태

  Card Reader의 필수 기능(Compile, Llink, Load)들을 Memory에 상주시키는 OS를 최초로 사용

- MS-DOS는 Batch Processing System

<br>

### 다중프로그래밍 시스템, Multiprogramming System

- Memory에 여러 개의 Program들을 적재하여 처리

- 한 번에 한 프로그램만을 적재하여 사용하던 일괄처리시스템보다 효율적으로 컴퓨터를 사용

- CPU Scheduling으로 인해 Program의 처리 순서 변경이 가능하며,

  속도가 느린 I/O 장치가 끝날 때까지 CPU가 기다리(Idle)지 않아도 다른 Program을 실행할 수 있음

- Mamory Management로 인해 Memory의 빈 공간에 Program을 할당하기가 용이

- 다른 Program들 끼리 침범하지 않도록 보호

<br>

### 시공유 시스템, Time-Sharing System

- 단말기(모니터+키보드)를 이용하여 컴퓨터 1대를 여러 사용자가 공유

- 강제 절환

  컴퓨터의 사용을 N명의 사용자가 번갈아가며 사용

- 시분할 시스템

  강제 절환을 하는데, N명의 사용자가 컴퓨터를 짧은 시간 동안 번갈아가며 사용

  ex) 1초를 3명이 나누어 쓸 때, 1인 당 약 30회의 기회를 부여

  컴퓨터의 빠른 속도로 인해 사용자들은 자기 혼자 컴퓨터를 쓰는 것 처럼 느낄 수 있다.

- 대화형 시스템(Interactive System)

  모니터와 키보드를 사용하여 컴퓨터와 대화하듯 사용

- 가상 메모리, Virtual Memory

  사용자가 많아짐에 따라 Main Memory가 부족하므로,

  Hard Disk를 Main Memory인 것 마냥(실제로는 그렇지 않음) 사용

- Process 간 통신

  Data를 전송하여 공유

- 동기화

- Windows, Linux, OSX, Android, iOS 모두 TSS 방식

<br><br>

# 1-3. 고등 운영체제

고등 컴퓨터 구조(Advanced Computer Architectures)가 등장하며, 고등 운영체제 또한 등장

- 폰 노이만의 컴퓨터 구조(일반적인 컴퓨터 구조)

  (하나의 CPU) + (하나의 Memory)

<br>

### 다중 프로세서 시스템, Multiprocessor System

(하나의 Memory)를 (여러 CPU)들이 병렬적으로, 강결합하여 사용

- 다음과 같이도 불림

  - **병렬 시스템, Parallel System**
  - **강결합 시스템, Tightly-Coupled System**

- 장점 3가지

  - Performance

  - Cost

  - Reliability

    고장난 CPU를 다른 CPU가 대체 가능

- 다중 프로세서 운영체제, Multiprocessor OS

<br>

### 분산 시스템, Distributed System

분산되어 있는 여러개의 PC Set((하나의 CPU) + (하나의 Memory))들을 LAN(근거리 통신망)을 통해 연결

- 다음과 같이도 불림
  - 다중 컴퓨터 시스템, Multi-Computer System
  - 소결합 시스템, Loosely-Coupled System
- 장점 3가지
  - Performance
  - Cost
  - Reliability
- 분산 운영체제, Distributed OS

<br>

### 실시간 시스템, Real-Time System

빠르기만 한 것이 아닌, Deadline을 준수하도록 작업을 수행

- 시간 제약 : Deadline

  CPU Scheduling으로 우선 순위를 적용

- 공장 자동화(FA), 군사, 항공, 우주 등의 분야에서 사용

- 실시간 운영체제, Real-Time OS(=RTOS)

