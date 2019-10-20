# 3-2. 임계 구역, Critical Section

임계 구역이란 공통 자원(variables, table, file, etc)을 update하는 구간(코드)을 의미하며, 3-1-1 예제에서의 `balance = balance + 10000` 과 `balance = balance - 10000` 코드들이 임계 구역에 해당됩니다.

<br>

<br>

## 3-2-1. 임계 구역 문제의 해결

임계 구역에서 발생하는 문제를 해결하기 위한 해결책은 다음 세 가지가 있습니다.

<br>

### 상호 배타, Mutual Exclusion

임계 구역에는 하나의 쓰레드만 진입할 수 있습니다.

<br>

### 진행, Progress

진입의 결정(어느 쓰레드가 먼저 공통 작업을 업데이트 할 것인가)은 유한 시간 내에 이루어져야 합니다.

<br>

### 유한 대기, Bounded Waiting

어떠한 쓰레드라도 유한 시간 내에 진입할 수 있어야 합니다.

