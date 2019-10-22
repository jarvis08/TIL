# 6-3 디스크 스케쥴링, Disk Scheduling

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

First-Come, First-Served 스케쥴링입니다. 가장 간단하며 공정한 방법으로, **디스크 큐(Disk Queue)**에 쌓이는 요청들을 시간 순서대로 처리합니다.

<br>

### 예시

Disk_Queue = [98, 183, 37, 122, 14, 124, 65, 67]

headers'_cylinder = 53

위의 Disk Queue 내용으로 탐색 대기열이 있다고 해 봅시다. FCFS 스케쥴링 방법은 현재 헤더의 위치인 53과 무관하게, 0번 인덱스 부터 차례로 방문합니다. 따라서 헤더의 이동 거리는 236 Cylinders가 됩니다.

<br>

<br>

## 6-3-2. SSTF Scheduling

