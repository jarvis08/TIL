# 1-1. 운영체제의 정의

## 1-1-1. 운영체제의 정의와 목적

### OS 정의

Control Program for computer which manage resources

<br>

### OS 목적

- **Performance**
- **Convenience**

<br>

<br>

## 1-1-2. 운영체제의 사용

### 컴퓨터 구조

- Process

- Memory

  - ROM

    - Main Memory의 극히 일부의 공간을 차지

    - 비휘발성 메모리

    - 컴퓨터를 Power-on 시 POST와 Boot Loader를 실행하는 코드를 저장

      - POST(Power-On Self-Test)

        어떤 H/W들이 연결되어 있는지 확인

      - Boot Loader

        Hard Disk에서 OS를 찾아내고, **RAM 영역으로 이동시켜 적재(Booting)**

      - Power-On일 경우 OS는 메모리에 상주(Resident)

  - RAM

    - 예시) Smart Phone의 Flash Memory
    - Main Memory의 대부분의 공간을 차지
    - 휘발성 메모리이므로 Power-Off 시 저장 내용이 모두 증발
    - OS의 코드가 부팅되는 메모리이며, 컴퓨터 실행 후 수행하는 프로그램 등의 내용을 저장

- Disk(보조기억장치), Network, Mouse, Keyboard, Speaker, GPS 등

- Program 내장형 컴퓨터

  Power-Off일 때에도 Program을 Memory에 내장

  - Program

    **Process**에게 명령하는 **Memory**의 코드(명령어)들을 **Instruction**이라고 하며,

    Instruction들의 집합을 Program이라고 한다.

<br>

### 커널과 쉘

- 커널, Kernal

  Shell이 해석 및 번역한 사용자의 명령을 **수행**, H/W를 조작

- 쉘, Shell(Command Interpreter)

  사용자로부터 받은 명령을 OS의 Kernal이 이해할 수 있도록 **해석 및 번역**

<br>

### 단계 별 예시

1. Applications

   사용자가 사용하는 Programs

2. Operating System

   Process Management, Memory Management, File Management, I/O Mgmt, Network Mgmt, Security Mgmt 등

3. Hardware

   CPU, Memory, Disk, Monitor, Printer 등

<br>

<br>

## 1-1-3. 컴퓨터의 규모별 분류

### 과거

1. Supercomputer

2. Mainframe

   수백대 단말기 규모

3. Mini

   수십대 단말기 규모

4. Micro



### 현재

1. Supercomputer

2. Server

   Web, Video 등

3. Workstation

4. PC

5. Handheld

   Laptop, Smart Phone, Tablet

6. Embedded