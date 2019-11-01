# 2-3. CPU Scheduling Algorithms

## 2-3-1. First-Come, First-Served (FCFS)

Non-Preemptive 방식이며, 간단하며 공정한 방법입니다. 프로세스가 생성된 순서대로 작업을 시작하여 끝마칩니다.

- AWT(Average Waiting Time)를 구하는 예제

  `Process` - `Burst Time`의 관계가 다음과 같을 때, AWT를 구하라.

  ```
  P1 - 24 msec
  P2 - 3 msec
  P3 - 3 msec
  ```

  `AWT = (0 + 24 + 27) / 3 = 17 msec`

<br>

### 호위 효과, Convoy Effect

위의 예시와 같이, 긴 작업 시간을 필요로 하는 프로세스를 다른 프로세스들이 따라다니며 대기하는 효과를 말하며, FCFS 알고리즘의 단점입니다.

<br>

<br>

## 2-3-2. Shortest-Job-First (SJF)

Preemptive 방식과 Non-Preemptive 방식이 모두 가능한 스케쥴링입니다. 이 스케쥴링 방법은 Burst Time이 가장 짧은 프로세스부터 실행합니다. 가장 이상적인 스케쥴링이지만, Burst Time을 예측하는 것은 현실적으로 불가능하며, 가능하다 해도 예측에 따른 Overhead도 크게 발생합니다.

- FCFS의 AWT를 구하는 예제를 SJF 스케쥴링한다면, 결과는 다음과 같습니다.

  `AWT = (6 + 0 + 3) / 3 = 7 msec`

<br>

<br>

## 2-3-3. Shortest-Remaining-Time-First

최소 잔여 시간이 되도록 고려하는 것이며, 프로세스의 Arrival Time(모든 프로세스가 시점 0부터 대기하는 것이 아니며, 각각의 Arrival Time 부터 대기)을 고려하여 SJF를 수행하는 스케쥴링입니다.

- `Process` - `Arrival Time` - `Burst Time(sec)` 이 다음과 같을 때의 AWT를 구하시오.

  ```
  P1 - 0 - 8
  P2 - 1 - 4
  P3 - 2 - 9
  P4 - 3 - 5
  ```

  `AWT = {(0 + 9) + 0 + 15 + 2} / 4 = 6.5 msec`

<br>

<br>

## 2-3-4. Priority Scheduling

전형적으로 정수 값이며, 낮을 수록 높은 우선 순위(Priority)임을 뜻하는 Priority 값에 의해 스케쥴링합니다. Priority 종류의 예시는 아래와 같습니다.

- Internal

  Time limit, Memory Requirement, I/O to CPU burst, etc.

  (I/O to CPU burst: CPU burst가 짧으며, I/O burst가 길 수록 우선 순위가 높다)

- External

  Amount of funds being paid, Political Factors

  ex) 앱 서비스에서, 유료 사용자들은 무료 사용자 보다 우선 순위가 높다.

<br>

### Starvation Problem

Priority Scheduling에서 발생하는 **문제점**은 **Starvation**(기아)입니다. Starvation는 우선 순위가 낮은 프로세스들에서 발생합니다. 우선 순위가 낮은 프로세스의 경우 새로운 프로세스 보다도 우선 순위가 밀려, 대기만 하게 되는 Indefinite Blocking 상황이 발생할 수 있습니다. Starvation의 **해결 방법**으로는 **Aging**이 있는데, aging은 대기 시간이 길어질 수록 우선 순위를 점차 늘려가는 방법입니다.

<br>

<br>

## 2-3-5. Round-Robin (RR)

오직 Preemptive 방식으로만 사용 가능하며, Time-Sharing System(시분할/시공유 시스템)에서 주로 사용하는 방법으로, 시간을 쪼개어 돌아가며 작업을 수행합니다. 여기서 쪼개지는 시간의 단위를 **Time Quantum**(시간 양자), **Time Slice**라고 하며, 주로 `10~100msec`을 사용합니다.

<br>

### Time Quantum

이 스케쥴링 방식의 성능은 Time Quantum의 크기로 결정됩니다. Time Quantum이 무한에 가까울 경우 FCFS 스케쥴링과 동일한 역할을 수행하며, 0에 가까울 경우 **Process Sharing**을 하는 것과 거의 동일합니다. Project Sharing이란 모든 프로세스들이 거의 동시에 작업되는 환경을 말하며, **Dispatcher**의 작업 비용이 증가하여 **Context Switching Overhead**가 발생하게 됩니다.

<br>

<br>

## 2-3-6. Multilevel Queue Scheduling

여러 개의 Ready Queue들을 사용하는 스케쥴링으로, 프로세스들을 그룹으로 나누어 그룹 당 하나의 Queue를 사용하도록 합니다. 프로세스 그룹들의 예시는 다음과 같습니다.

- System Processes

  OS 내부에서 발생하는 프로세스(Memory Allocation)

- Interactive Processes

  사용자와의 Interaction(마우스 포인터 이동, 클릭, 키보드 입력 등)

- Interactive Editing Processes

  워드 프로세서와 같은 타입핑이 자주 사용되는 프로세스

- Batch Processes

  사용자 Interaction이 아닌 대표적인 프로세스(설치 마법사 실행 및 설치 작업)

<br>

### 특징

Single Ready Queue가 아닌, **Several Separate Queues**를 사용하며, 각각의 **Queue 별로 절대적인 우선 순위**가 존재할 수 있습니다. 우선 순위로 Queue를 CPU에 할당하는 것이 아니라면, **CPU time**을 각 Queue에 차등 배분할 수 있습니다.

각 Queue안의 Process들에는 **Queue 별로 독립된 스케쥴링 정책**이 사용될 수 있습니다. 예를 들어, System Process에는 FCFS 스케쥴링을, Batch Processes에는 Round Robind을 적용하는 등 다르게 스케쥴링할 수 있습니다.

<br>

<br>

## 2-3-7. Multilevel Feedback Queue Scheduling

복수 개의 Ready Queues를 운영하고 있을 때, 프로세스를 **다른 Queue로 점진적으로 이동시킬 수 있습니다**. 너무 많은 CPU Time을 사용 시, 다른 Queue로 이동 시킬 수 있으며, Starvation 우려 시 우선 순위가 높은 Queue로 이동시키기도 합니다.

<br>

<br>

## 2-3-8. 프로세스의 생성과 종료

프로세스는 프로세스에 의해 생성됩니다. 프로세스의 종류로는 부모 프로세스(**Parent Process**)와 자식 프로세스(**Child Process**)가 있으며, 같은 부모를 둔 자식 프로세스들은 서로 **Sibling Process**입니다.

<br>

### Process Tree

운영체제가 Booting을 완료한 후, `init`이라는 프로세스를 자동으로 생성합니다. 이후, `init` 프로세스는 사용자의 명령에 따라 다른 프로세스들을 생성하게 되며, 프로세스 트리(**Process Tree**)를 형성합니다.

<br>

### Process Identifier (PID)

전형적으로 정수이며, 생성되는 순서대로 프로세스에 고유한 ID를 부여합니다.

PPID(Parent Process Identifier)는 부모의 PID를 의미합니다.

<br>

### 프로세스를 생성하고 종료하기

- `fork()`: 새로운 프로세스를 생성
- `exec()`: 생성한 프로세스를 메모리에 로드하여 적재
- `exit()`: 프로세스를 종료

프로세스 종료 시, 해당 프로세스가 보유했던 모든 자원(메모리, 파일, 입출력 장치)을 운영 체제에게 반환합니다.

