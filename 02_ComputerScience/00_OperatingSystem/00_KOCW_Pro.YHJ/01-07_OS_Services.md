# 1-7. 운영체제 서비스, OS Services

## 1-7-1. 프로세스 관리, Process Management

프로세스, Process: **메모리에서 실행 중인 프로그램**, Program in execution

<br>

### 주요 기능

- 프로세스의 생성(**Creation**), 소멸(**Deletion**)
- 프로세스 활동 일시 중지(**Suspend**), 활동 재개(**Resume**)
- 프로세스 간 통신(**IPC**, Inter-Process Communication)
- 프로세스 동기화(**Synchronization**)
- 교착 상태 처리(**Deadlock Handling**)
  - Deadlock: 다수의 프로세스가 실행되던 중 충돌이 발생한 상태

<br>

<br>

## 1-7-2. 주기억장치 관리, Main Memory Management

### 주요 기능

- 프로세스에게 메모리 공간을 할당(**Allocation**)

- **메모리의 어느 부분이 어느 프로세스에게 활당되었는지 추적 및 감시**

- 프로세스종료 시 메모리 회수(**Dallocation**)

- 메모리의 **효율적인 공간 사용** 유도

  i.g., 사용 가능하지만 사용되지 않는 메모리 공간을 관리

- **가상 메모리** 관리

  실제 물리적 메모리보다 큰 용량을 사용하도록 조치

<br>

<br>

## 1-7-3. 파일 관리, File Management

**Track**/**Sector**로 구성된 **디스크**를 **파일**이라는 논리적 관점으로 관리

Hard Disk는 자성을 띈 판 위에 **Track**이 깔려 있으며, Track은 여러 **Sector**들로 구분된다. **Coil**이 감긴 **Header**를 통해 Sector들을 N/S극으로 **자화**시켜 데이터를 저장 및 복사한다. 자극하는 Sector들을 묶어 **Block** 단위로 관리한다.

<br>

### 주요 기능

- 파일의 생성(**Creation**)과 삭제(**Deletion**)

- 디렉토리(**Directory**)(or Folder)의 생성과 삭제

- 파일에 대한 **기본 동작** 지원

  ```open, close, read, write, create, delete```

- Track/Sector와 파일간의 매핑(**Mapping**)

- 백업(**Backup**)

<br>

<br>

## 1-7-4. 보조 기억 장치 관리, Secondary Storage Management

- 보조 기억 장치(Secondary Storage)의 예시

  Hard Disk, Flash Memory(in Smart Phone)

<br>

### 주요 기능

- **빈 공간 관리, Free Space Management**

  (빈 Block들)

- 저장 공간 할당, **Storage Allocation**

  어느 Block에 할당할 것인가?

- 디스크 스케쥴링, **Disk Scheduling**

  디스크 공간에 Block들이 흩어져 있을때, Block들의 Head들을 어떻게 하면 효율적으로 찾아다니며 읽을 수 있겠는가?

<br>

<br>

## 1-7-5. 입출력 장치 관리, I/O Device Management

### 주요 기능

- 장치 드라이브, Device Drivers

- 입출력 장치 성능 향상

  - Buffering

    입출력 장치에서 읽었던 내용을 메모리에 유지하여, 같은 내용이 다시 호출됐을 때 빠른 속도로 반환

  - Caching

    Buffering과 유사

  - Spooling

    Memory 대신 Hard Disk를 중간 매체로 사용하여 효율을 증가시킨다.

    i.g., 프린터에서 작업할 내용을 계속해서 Memory에 두는 것이 아니라, 비교적 덜 귀한 Hard Disk에 저장



