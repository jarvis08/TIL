# 2-2. CPU Scheduling Criteria

CPU Scheduling에는 **Preemptive** 방식과 **Non-Preemptive** 방식이 있습니다.

- Preemptive

  우선순위에 따라 CPU 할당을 결정 및 변경합니다.

- Non-Preemptive

  이미 CPU가 어떤 프로세스를 작업하고 있다면, 해당 프로세스가 종료되어야만 다음 프로세스를 작업할 수 있습니다.

<br>

### CPU Utilization, 이용률

다음 두 가지로 프로세스를 스케쥴링할 수 있다고 가정해 봅시다.

1. `P1 - P2 - P3`
2. `P3 - P1 - P2`

만약 1번 과정은 CPU를 100% 활용하여 작업할 수 있으며, 2번 과정이 80%만을 사용할 수 있다면, CPU 이용률 측면에서 봤을 때 1번 과정이 더 효율적입니다.

<br>

### Throughput, 처리율

단위 시간 당 몇 개의 작업을 끝낼 수 있는가? (`job/sec`)

<br>

### Turnaround Time, 반환 시간

프로세스가 Ready Queue에 등록되는 시점으로부터 삭제되기 까지 걸리는 시간(`sec`)입니다.

<br>

### Waiting Time, 대기 시간

Ready Queue에서 CPU를 할당 받기 위해 기다리는 총 시간(`sec`)입니다.

<br>

### Response Time, 응답 시간

Interactive System에서 주요하게 다뤄지는 개념입니다. Interactive System에서 사용자와 컴퓨터가 신호를 주고받을 때, 사용자가 컴퓨터로 부터 첫 응답을 받는데 까지 걸리는 시간입니다.

