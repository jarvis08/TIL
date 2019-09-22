# 1-3. 고등 운영체제, Advanced OS

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