# Operating System Intro

참고 자료: 고려대학교 최린 교수님, KUOCW 운영체제 강의

---

## Lecture Overview

1. OS Overview

2. Process

   실행 중인 프로그램

3. Thread

   가벼운 Process

   하나의 Process는 여러 개의 Thread를 보유 가능

4. Mutual Exclusion and Synchronization

   (Mutual Exclusion)여러 Process가 IO 장비(e.g., Printer)와 같은 None Shared Resource를 동시에 사용하지 않도록 Protect

5. Deadlock and Starvation

   - Deadlock

     Process끼리 서로 Resource를 점유하려 시도하여 충돌이 된 상태

   - Starvation

     Resource 사용을 계속 대기중인 상태

6. Memory Management

7. Virtual Memory

8. Uniprocessor Scheduling

   어느 Process가 CPU(Multi-Core의 경우 여러 일 가능)를 사용할 지 결정하는 알고리즘

   e.g., Uni-process Scheduling

9. Multiprocessor and Realtime Scheduling

10. IO

    Disk Scheduling 성능 최적화

    Disk는 속도가 느리다

11. File Management

12. Virtual Machine

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

    Individualy Seperated, Equal Level Works

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

- CISC, Complex Instruction Set Computer

  80년대 이전까지 복잡하지만 정교한 Instruction Set

  프로그램 설계가 용이하지만, 속도가 느리다.

  - Each instruction is complex

  - A large number of instucutions in ISA

  - Architectures until mid 80's

    e.g., x86

- RISC, Reduced Instruction Set Computer

  - A small number of Instructions in ISA
  - Load-store architectures
    - Computations are allowd only on registers
    - Data must be transferred to registers before computation
  - Most architectures built since 80's

- Word

  Default data size for computation

  - Size of GPR & ALU data path depends on the word size

    - GPR

      General purpose (integer) register

    - ALU

      Arithmetic and logic unit

  - The word size determines if a processor is a 8b, 16b, 32b, or 64b processor

- Address(Pointer)

  Points to a location in memory

  - Each address points to a byte(byte addressable)
    - 32bit address가 있을 때, `2^32` bytes(4GB) 만큼 address 가능
    - 256MB memory가 있을 때, 최소 28bit address 필요(`2^28` = 256MB)

- Caches

  Faster but smaller memory close to processor

  - SRAMs에 설치되어 있어 속도가 빠르지만, 그 만큼 제한된 용량을 보유

- Interrupt

  - A mechanism by which I/O devices may interrupt the normal sequencing of the processor
  - Provided primarily as a way to improve processor utilization since most I/O devices are much slower than the processor
  - More formally, interrupt can be defined as below
    - Forced transfer of control to a procedure(handler) due to external events(interrups) or due to an erroneous condition(exceptions)
    - External interrupt is caused by external evnets(I/O devices) and asynchronous
    - Exceptions are caused by processor internally at erroneous condition

---

## Evolution of Operating Systems

1. Serial Processing
2. Simple Batch Systems
3. Multiprogrammed Batch Systems
4. Time Sharing Systems

### Serial Processing

- Earliest computers

  - No OS until mid 1950s

    Programmers interacted directly with the computer H/W

  - Computers ran from a console with,

    display lights, toggle switches, some form of input device, printer

- Problems

  - Scheduling

    사용하기 위해 연구자들의 사용 예약 요구

  - Setup time of program to run

    - Compile
    - Link
    - Load require mounting tapes
    - Setting up card decks

  - Extemely expensive

- 인물

  - Alan Turing's

    Computer Machine 최초의 이론인 Turing Machine을 제안

    - Bombe

      컴퓨터에 가까운 독일군 암호 해독 기계

    - Colossus

      Turing이 직접 발명에 참여한 것은 아니나, Bombe를 기초로 만든 세계 최초의 전자식 컴퓨터

      ENIAC 발명 이전의 기계이며, Disclosed 되지 않았었다.

    - Turing test

      설명 생략

  - The Von Neumann Machine

    최초의 현대식 컴퓨터 개발

    - IAS

      IBM comercial computer의 Base

### Simple Batch Systems

- Moinitor

  먼저 제출한 프로그램 부터 차례대로 수행(FIFO)

  - User submits the job on cards or tpae to a computer operater, who batches them together sequentially and places them on an input device
  - Monitor is a resident S/W in main memory
  - Monitor reads in jobs one at a time from the input device
  - The current job is placed in the user program area
  - The control is passed to the job
  - When the job is completed, it returns control to the monitor
    - User no longer has direct access to the processor

- History

  - The 1st batch OS was developed by GM in the mid-1950s for use on IBM 701
  - By the early 1960s, a number of vendors developed batch OS for their computer systems

- Problems

  - Processor is often idle

    - Even with automatic job sequencing

    - I/O devices are slow compared to processor

      ```
      15ms - Read one record from file
      1 ms - Execute 100 instructions
      15ms - Write one record to file
      Total = 31ms
      ```

### Multiprogrammed Batch System





















