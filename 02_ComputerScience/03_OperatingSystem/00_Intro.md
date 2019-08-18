# Operating System Intro

참고 자료: 고려대학교 최린 교수님, KUOCW 운영체제 강의

---

## Lecture Overview

1. OS Overview
2. Process
3. Thread
4. Mutual Exclusion and Synchronization
5. Deadlock and Starvation
6. Memory Management
7. Virtual Memory
8. Uniprocessor Scheduling
9. Multiprocessor and Realtime Scheduling
10. IO
11. File Management
12. Virtual Machine

- Process

  실행 중인 프로그램

- Thread

  가벼운 Process

  하나의 Process는 여러 개의 Thread를 보유 가능

- Mutual Exclusion

   여러 Process가 IO 장비(e.g., Printer)와 같은 None Shared Resource를 동시에 사용하지 않도록 Protect

- Deadlock

  Process끼리 서로 Resource를 점유하려 시도하여 충돌이 된 상태

- Starvation

  Resource 사용을 계속 대기중인 상태

- Memory

  Main Memory와 Virtual Memory

- Scheduling

  어느 Process가 CPU(Multi-Core의 경우 여러 일 가능)를 사용할 지 결정하는 알고리즘

  e.g., Uni-process Scheduling

- IO

  Disk Scheduling 성능 최적화

  Disk는 속도가 느리다

---

## 1. Operating System Overview

- What is computer
  - Hardware가 존재 하는가
  - Software를 실행하는가

### Computer Hardware and Software Infrastructure

- HW - Memory, I/O Devices, Networking

  SW - Operating System, Library/Utility, Application

- Operating System

  Program that provides convenient services for application programs by managing hardware resources,

  and acts as an interface between applications and the computer hardware

  1. ISA, Instruction Set Architecture

     Define the interace between SW and HW

     - A set of machine language intruction
  
   - Both application programs and utilites can access the ISA directly
  
   - Hardware를 제어하기 위해 사용하며, OS를 이용하여 Instruction Set을 편리하게 사용 가능
  
   - x86은 intel이 사용하는 32bit processer에서 사용하는 명령어 set(=IA32)
  
     i3~i9 등은 **x64 ISA**를 사용
  
     ***같은 ISA**를 사용하더라도, **내부 설계(Micro Architecture) 구조**에 따라 CPU의 처리 속도가 다르다.
  
   - 보통 우리가 사용하는 `.exe` 확장자의 binary file들은 Intel의 HW language를 사용
  
2. ABI, Application Binary Interface
  
   Define the system call interface to OS
  
   - OS의 Kernal이 제공하며, 이를 이용하여 **Library** 혹은 **Utility**를 제작
  
3. API, Application Programming Interface
  
   Define the program call interface to system services.
  
   System calls are performed thorugh library calls.
  
     - Enables application programs to be ported easily, through remopilation to other systems that support the same API
    
     - Library가 제공하는, Application Program들을 제작할 때 사용하는 편리한 Interface

### OS의 또 다른 정의

- OS

  A layer of SW between the application program and the HW
  - Provide applications with simple and uniform interface to manipulate complicated and often widely dirrerent low-level hardware devices
  - Protect the hardware from misuse by runaway applications
  - Share the hardware resource from multiple processes/users

  - Use abstractions such as processes, virtual memory, and files to achieve both goals

- Layer View of a Computer System

  - SW

    OS level is lower than App Programs

    - Application Programs
    - Operating System

  - HW

    Individualy Seperated Works

    - Processor, Main Memory, I/O Devices

- Abstractions provided by an OS

  Processes > Virtual Memory > Files

  - Processor(Processes)

  - Main Memory(Virtual Memory)

    프로그램이 차지하고 있는 공간이며, D-Ram은 Physical Memory이다.

    Physical Memory에서는 중요한 Process를 담당

  - I/O Devices(Files)

---

### Terminology

- Microprocessor

  a single chip processor

  e.g., Intel i7, Pentium 4, AMD Athlon, SUN Ultrasparc, ARM, MIPS, ...

- ISA, Instruction Set Architecture

  Defines machine instructions and visible machine states such as registers and memory

  Contains Sets of HW Language

  e.g., x86(IA32), x64(IA64)

  - x86 : 2**32 byte 까지 표현이 가능하므로, 4GB 이상의 프로그램은 사용 불가

    Sequential Circuit으로 생각 했을 때 (2**32) * 8 Memory State가 존재하는 것이며,

    CPU는 Memory에서 받아온 정보를 32개(32bit)의 Register에 32 * 32 State에 저장

    - 컴퓨터의 명령

      Memory로부터 data를 load하여 계산한 후 store를 통해 Memory에 계산한 data를 저장

      Memory의 data를 읽어서 화면에 출력

      - Load

        Memory의 정보를 Register로 가져옴

        가져왔던 Register 정보를 32bit 값으로 전환

        Memory State의 변화는 없으며, 단순한 State Transition

      - Add

        Register1과 Register2의 값을 더하여 Register3에 저장

        Machine State의 변경

      - Store

        Register의 32bit 값을 Memory의 4byte 공간에 저장

- Microarchitecture

  - Implement hardware according to the ISA
    - Pipelining, Caches, Branch Prediction, Buffers

    - 80386, 80486, Pentium, Pentimu Pro, Pentium 4 are the 1st, 2nd, 3rd, 4th, 5th implementation of x86 ISA

      동시 명령어 처리 개수가 점차 증가

      386의 경우 Pipelining을 지원하지 않았다.

      Cahce는 Memory가 아닌 CPU에서 data를 가져오기에 속도가 빠르다.

  - Invisible to Programmers

    Programmers program Pentium 4 as same as 486 processor

    

  