# 2-4. 쓰레드, Thread

쓰레드는 프로그램 내부의 흐름이자 맥이라고 표현할 수 있습니다. 실질적으로, CPU의 Context Switching 단위는 Process가 아니라, Thread입니다. 예를 들자면, P1의 T3 작업에서 P2의 T2 작업으로 Switching 하게 됩니다. 그리고 CPU의 Switching에는 Process Switching과 Thread Switching이 있습니다.

<br>

<br>

## 2-4-1. 다중 쓰레드, Multi Threads

한 프로그램에 2개 이상의 맥이 있을 경우입니다. 맥이 빠른 시간의 간격으로 스위칭 된다면, 여러 맥이 동시에 실행되는 것 처럼 보입니다. 하지만 이는 Simultaneous(동시의)한 것이 아니라, Concurrent(스위칭이 빠름)하다고 표현합니다.

다중 쓰레드의 예시는 다음과 같습니다.

- Web Browser
  - 화면을 출력하는 쓰레드
  - 데이터를 읽어오는 쓰레드
- Word Processor
  - 화면을 출력하는 쓰레드
  - 키보드 입력을 받는 쓰레드
  - 철자/문법의 오류를 확인하는 쓰레드

하나의 프로세스는 기본적으로 **Single Thread Program**이며, 여러 개의 쓰레드가 있을 경우 **Multi-Thread Program**이라고 합니다.

<br>

<br>

## 2-4-2. 쓰레드의 구조

하나의 프로세스와 그 프로세스의 여러 쓰레드들이 있을 때, 쓰레드들 간에 공유되는 사항들이 있으며, 그렇지 않은 것들이 있습니다.

<br>

### 공유

- 메모리 공간

  **Code**, **Data**

- 자원

  file, I/O, ...

<br>

### 비공유

- 개별적인 PC, SP, Register, 메모리의 **Stack**

