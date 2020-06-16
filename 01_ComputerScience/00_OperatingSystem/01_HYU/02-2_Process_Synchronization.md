# Process Synchronization

공유하고 있는 데이터에 concurrent한 접속 및 작업은 data inconsistency를 유발할 수 있다. 이는 **Race Condition**이라고 표하며, race condition은 **여러 process들이 같은 데이터에 접근해서 작업**을 하고, **마지막으로 `write`하고 끝낸 process의 작업 내용만 저장**되는 현상이다. 이를 방지하기 위해서는 **Synchronization**이 필요하다.

### Critical Section

여러 process들이 있을 때, 모든 process들이 동일한 shared data에 대해 작업하는 코드 구간을 가지고 있다. 이러한 **race conditiond을 유발할 수 있는 코드 구간**을 critical section이라고 한다.

### Solution의 조건

- Mutual Exclusion (흔히, mutex)
  - Shared data에 대해 하나의 process가 작업중일 때, 다른 어떤 process도 접근하지 못함
  - 확실하지만, 불필요한 경우까지 막을 수 있으므로 다른 추가 조건이 필요
- Progress
  - 과도하게 막아서 아무도 쓰지 않는 상황에서 접근하지 못하는 상황을 방지
- Bounded Waiting
  - Starvation을 방지

### Solution Algorithms

- Swap-turn
  - 무조건 번갈아 가며 사용
  - Shared data에 대해 $p_0$가 사용을 마친 후, $p_0$가 다시 shared data를 사용하려면 $p_1$이 사용하기를 기다려야 한다.
  - Mutex 만족
  - Progress는 불만족
    - $p_1$이 사용할 일이 없는데, $p_0$는 사용할 수 있는 상황임에도 불구하고 기다리기만 하는 상황이 발생
- Flag
  - 상태로 사용 의사를 표현
  - Mutex 만족
  - Progress는 불만족
    - `flag[p_0]=flag[p_1]=true`로, 둘 다 사용하고 싶다면?
- **Peterson's Algorithm**
  - (Swat-turn) + Flag
    - `Flag[상대]==true`, `turn==상대` 일 경우에 사용 불가
  - Mutex/Progress/Bounded Waiting 모두 만족
  - Busy Waiting
    - Process는 waiting 상태에서도 CPU를 사용

### Synchronization Hardware

Atomic = non-interruptable

위 atomic의 성질을 이용한 atomic H/W의 예시 두 가지는 다음과 같다.

- atomic H/W의 instruction인 `Test-and-Set`
  - 특정 작업을 다 수행하기 전까지, 다른 작업을 수행하지 않도록 **회로**에 구워진 함수
- atomic H/W의 instruction인 `Swap`
  - `lock`. `key` 두 가지 변수를 swap하여 key가 유입될 때 까지 진행중인 process 만을 실행

### Semaphore

Semaphore는 synchronization 문제를 Software 상에서 해결하기 위한 방법이다. Semaphore는 Peterson's algorithm의 개념을 사용함과 동시에, busy waiting을 해결하기 위해 `block`과 `wakeup()`을 추가적으로 사용한다.

- `S`
  - Semaphore 변수 (INT)
  - 초기값 = 0
  - `S` 값은 대기 중인 작업의 개수를 표현
- `P(S)`
  - Semaphore 값을 (-1)
  - block
- V(S)
  - Semaphore 값을 (+1)
  - `wakeup(P)`

Critical Section의 길이가 길 경우 Block-wakeup이 유리하며, 길이가 짧은 경우 busy-wait의 길이가 짧아지면서 busy-wait이 보다 유리하게 된다.

### Critical Section Problem in OS

- Kernel 실행 도중에는 CPU Scheduling이 발생하지 않도록 조치
- Kernel 전체를 하나의 큰 Critical Section으로 처리하여, 오직 하나의 CPU만이 kernel을 다룰 수 있다.
  - 너무 작업 성능이 떨어짐
- Kernel 변수 하나 하나에 semaphore를 적용
  - 너무 많은 ciritical section