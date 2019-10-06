# 2-1. 프로세스, Process

Process(= Task, Job)는 현재 실행되고 있는 프로그램(Program in execution)을 말하며, 다음의 정보들을 가진다

- Code(Text), Data, Stack

- PC, Program Counter

  주소 정보

- SP, Stack Point

- Register

- etc

<br>

## 2-1-1. 프로세스의 상태

- new

  Memory에 올라온 직후의 상태

- ready

  초기화 완료 및 실행 가능한 상태

- running

  CPU가 실제로 실행하고 있는 상태

- waiting

  CPU 사용이 현재 불필요하거나, 미뤄진 상태

- terminated

<br>

### 프로세스 상태 천이도

![Process state transition diagram](../../assets/Process-state-transition-diagram.png)

> 출처: https://www.researchgate.net/figure/Process-state-transition-diagram_fig3_332546783

위 프로세스 상태 천이도는 Multi-Programming System에 해당하는 도표이며, Time-Sharing System의 경우 `running`에서 `ready`로 향하는 **Time Expired** 흐름이 추가적으로 존재한다. Time-Sharing System의 경우 사용자 별로 시간을 분배하여 CPU를 할당하며, Time Expired 발생 시 작업하던 프로세스를 `ready` 상태로 되돌린다.

<br>

<br>

## 2-1-2. 프로세스 제어 블록, Process Control Block(PCB)

Task Control Block(TCB)라고도 불리며, 프로세스에 대한 모든 정보를 보유한다. 보유하는 정보의 내용은 다음과 같다.

- Process state

  - running, ready, waiting, etc

- 주소 정보, Program Counter(PC)

  다시 `running` 상태로 돌아왔을 때 작업을 재개하려면 PC, register 등의 정보가 필요하다.

- registers

- MMU info

  - base
  - limit

- CPU time, CPU 사용 시간

- Process ID

- List of open files

  사용 중인 파일 목록

- etc

<br>

<br>

## 2-1-3. 프로세스 대기열(Queue)

### Job Queue

메모리 사용의 대기열이다.

### Job Scheduler

대기열의 프로그램들 중 **어떤 프로그램을 메모리에 적재할 지 결정하는 프로그램**이다.

- Job Scheduling은 빈번하게 일어나지 않기 때문에 **Long-term Scheduler**라고도 부른다.

  i.g., 메모리 제한 용량이 가득차면, Job Scheduler가 할 일이 없다.

<br>

### Ready Queue

CPU 사용의 대기열이다.

### CPU Scheduler

CPU 대기열 중 어떤 작업을 `running` 상태로 전환할 지 선택하는 프로그램이다.

- `running`과 `waiting`을 빠르게 결정하여 작업을 진행하기 때문에 **Short-term Scheduler**라고도 부른다.

  우리는 이 작업이 빠르게 일어나기 때문에 프로그램들이 동시에 실행되고 있다고 느낄 수 있다.

<br>

### Device Queue

I/O 장치와 Disk의 대기열이며, Device Scheduler가 관리한다.

<br>

<br>

## 2-1-4. 멀티프로그래밍, Multiprogramming

- 멀티 프로그래밍의 차수, Degree of Multiprogramming

  Main Memory에 몇 개의 프로세스가 적재되어 있는가?

<br>

### I/O-bound Process & CPU-bound Process

Job Scheduler는 Job Queue에서 다음 작업을 선택할 때, 다음에 설명할 **I/O-bound Process**와 **CPU-bound Process**를 적절히 배분하여  선택해야 한다.

- I/O-bound Process

  작업해야 하는 내용이 주로 I/O를 사용하는 프로세스이며, 아래의 예시가 있다.

  - Word Processor
  - 타자 연습 프로그램

- CPU-bound Process

  CPU(계산) 사용이 많은 프로세스이며, 기상청의 일기 예측 프로그램과 같은 프로그램이 그 예이다.

<br>

### Medium-term Scheduler

Medium-term Scheduler는 어떤 작업을 **Swapping**(Swap In/Out)할 지 결정한다.

<br>

### Swapping

**대화형 시스템(Interactive System)**은 대부분 **Time-Sharing System**이다. Time-Sharing System의 예시로는 Windows와 Linux가 있다. Time-Sharing System은 **사용자 별로 시간을 분배하여 CPU를 할당**한다.

운영체제는 **PCB를 계속해서 확인**하며, 만약 한 사용자가 자리를 작업을 멈추었을 경우, 장기간 동안 작업을 하지 않은 프로세스 및 사용자의 작업 내용을 **Hard Disk**로 **Swap Out**한다. 그리고 다시 작업을 시작한 경우 메모리로 **Swap In**하며, 여기서의 Hard Disk는 **Swap Device**의 역할을 한다. Hard Disk의 물리적 공간은 File System으로 사용되는 공간과 Swap Device로 사용되는 **Backing Store**로 나뉜다.

<br>

### Context Switching

`Process-1`에서 `Process-2`로, `Process-2`에서 `Process-3`로 이동하는 등의 전환을 `Context Switching`이라고 한다.

- **Scheduler**

  현재 프로세스에서 어떤 프로세스로 Context Switching할 것인지 결정한다.

- **Dispatcher**

  1. `Process-1`의 현재 정보를 OS 메모리 내부의 `PCB-1`에 저장
  2. 전환하려는 프로세스인 `Process-2`의 `PCB-2`를 복원하여 다음 데이터 값들을 변경
     - PC(Program Counter)
     - SP(Stack Point)
     - Register
     - MMU(Base, Limit)

- **Context Switching Overhead**

  프로세스간 전환하는 과정에서 발생하는 오버헤드를 말한다.

  오버헤드를 감소시키기 위해서는 C와 같은 High Level Language보다는 Assembly Language를 사용해야 한다.


