# 6-3. 디스크 스케쥴링, Disk Scheduling

디스크가 데이터를 읽어오는 시간인 **디스크 접근 시간**은 다음과 같습니다.

`Seek Time + Rotational Delay + Transfer Time`

- Seek Time

  헤더가 데이터가 위치한 섹터 및 블록을 찾고, 이동하는데 걸리는 시간

- Rotational Delay

  디스크가 회전하는데 걸리는 시간

- Transfer Time

  전자기장을 형성하여 데이터를 읽어내는데 걸리는 시간

**디스크 스케쥴링 알고리즘(Disk Scheduling Algorithm)**은 디스크 접근 시간을 최소화 하는 방법에 대한 알고리즘입니다.

<br>

<br>

## 6-3-1. FCFS Scheduling

**First-Come, First-Served Scheduling**입니다. 가장 간단하며 공정한 방법으로, **디스크 큐(Disk Queue)**에 쌓이는 요청들을 시간 순서대로 처리합니다.

<br>

### 예시

200의 cylinder disk가 있을 때,

- Disk_Queue = [98, 183, 37, 122, 14, 124, 65, 67]

- headers'_cylinder = 53

위의 Disk Queue 내용으로 탐색 대기열이 있다고 해 봅시다. FCFS 스케쥴링 방법은 현재 헤더의 위치인 53과 무관하게, 0번 인덱스 부터 차례로 방문합니다. 따라서 헤더의 이동 거리는 640 Cylinders가 됩니다.

<br>

<br>

## 6-3-2. SSTF Scheduling

**Shortest-Seek-Time-First Scheduling**입니다. SSTF 스케쥴링은 현재 헤더 위치를 기준으로, 가장 가까운 거리에 위치한, seek time을 최소로 하는 요청부터 처리합니다.

<br>

### 예시

200의 cylinder disk가 있을 때,

- Disk_Queue = [98, 183, 37, 122, 14, 124, 65, 67]

- headers'_cylinder = 53

헤더의 위치에서 가장 가까운 곳을 탐색하여 이동하므로, [65, 67, 37, 14, 98, ...]의 순서로 이동하며, 헤더의 총 이동거리는 236 Cylinders입니다.

<br>

### 문제점

현재 예시만을 봤을때, 헤더의 총 이동 거리는 FCFS에 비해 월등히 낮습니다. 하지만 현실의 대부분의 경우에는 지속적으로 Disk_Queue에 요청이 유입되며, 만약 특정 위치의 실린더들에 대한 요청들이 지속적으로 들어올 경우 **Starvation**이 발생할 수 있습니다. Starvation은 특정 처리가 계속해서 처리되지 않는 상황을 말합니다. 또한, 모든 경우에 FCFS보다 좋은 성능을 보인다고도 할 수 없습니다.

<br>

<br>

## 6-3-3. SCAN Scheduling

SCAN Scheduling은 이름처럼, 디스크를 계속해서 앞뒤로 스캔하는 방식입니다.

<br>

### 예시

200의 cylinder disk가 있을 때,

- Disk_Queue = [98, 183, 37, 122, 14, 124, 65, 67]

- headers'_cylinder = 53

헤더의 총 이동 거리는  236(=53+183) Cylinders이며, SSTF Scheduling 방식보다 적은 시간을 소요합니다.

<br>

<br>

## 6-3-4. SCAN Scheduling's Variants

SCAN Scheduling을 변형한 방법이 총 세 가지 있습니다.

- C-SCAN
- LOOK
- C-LOOK

<br>

### C-SCAN

만약 계속되는 실린더의 요청이 **연속균등분포(Uniform Distribution)**를 따른다고 가정해 봅시다. 연속균등분포를 따를 경우 실린더 공간에 고르게 요청되며, scheduling 방식에 의해 scan하는 헤더 주변이 가장 요청이 적은 지역일 것입니다. 헤더의 위치가 한쪽 끝일 경우, 요청이 가장 많은 지역은 그 반대쪽 끝입니다. 

이러한 가정에 입각하여 제안된 것이 **Circular SCAN**입니다. Circular SCAN은 **circular list**가 순환하듯, 헤더의 이동 방향이 하나입니다. **한쪽 끝에 도달**한 후 반대쪽 방향으로 다시 탐색을 하는 것이 아니라, **바로 반대 쪽 끝으로 이동하여 다시 같은 방향으로 탐색**합니다.

<br>

### LOOK

**SCAN Scheduling** 방식을 따르지만, 헤더가 **진행 방향의 마지막 요청 위치 까지**만 스캔합니다. 그러기 위해 움직이기 직전의 순간마다, 지속적으로 진행 방향으로의 요청이 있는가를 확인합니다.

<br>

### C-LOOK

**C-SCAN**의 방식을 따르지만, LOOK과 같이 **마지막 요청 위치까지만 스캔**합니다.

<br>

### Elevator Algorithm

**C-LOOK**의 움직임이 엘리베이터의 움직임과 동일하다 하여 **Elevator Algorithm**이라고도 불립니다.

