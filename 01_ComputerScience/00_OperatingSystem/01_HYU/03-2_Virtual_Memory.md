# Virtual Memory

Virtual Memory는 physical memory 공간을 더 효율적으로 사용하기 위한 개념이다. 즉, virtual memory를 사용하지 않아도 작동 자체에는 문제가 없다.

Virtual memory는 processor가 작업할 때, process의 일부만을 사용하여 작업한다는 점을 이용한다. 즉, processor가 필요로 하는 일부 데이터만 load해 두고, 나머지 데이터는 disk의 virtual memory에 저장해 둔다. 그로인해 얻을 수 있는 효과는 다음과 같다.

- 메모리 공간 활용의 효율이 높아진다.
- 동시에 더 많은 process들을 실행할 수 있다.

CPU는 process의 모든 정보가 load 되어 있다고 생각한 채 작업을 하게 되며, CPU가 load 되지 않은 데이터를 요구할 때에는 Demand Paging/Demand Segmentation 등을 이용하여 필요한 page/segment를 load하게 된다.

### Demand Paging

Demand Paging은 Lazy Swapper이다. Page를 다루는 swapper를 **Pager**라고 하며, pager는 필요하기 전까지 절대 page를 swap하지 않기 때문에 lazy swapper라고 한다.

Demand Paging은 MMU에 **valid bit**의 column을 추가하여 구현한다. 이는 상태 처리(on/off)로 요구되는 page가 physical memory 공간에 load 되어 있는가를 나타낸다.

valid bit이 **Invalid**라면, **Page Fault**가 발생하며, Invalid는 다음과 같은 의미를 가진다.

- Illegal
  - 해당 page는 메모리 주소 공간에 없다(잘못된 주소)
- not-in-memory
  - DRAM에 load되지 않았다.
- obsolete
  - Disk로 부터 온 데이터인데, 현재 disk의 값과 다르다
  - reload하여 최신화가 필요

모든 valid bit의 초기값은 invalid이다.

### Page Fault

Page Fault는 invalid한 page로 access를 시도하는 것이며, 이는 H/W(MMU) trap인 **Page Fault Trap**을 유발한다. 그리고 OS의 Trap Handler는 다음과 같은 절차로 page fatult를 다룬다.

1. 어떤 유형의 page fault인가
   - illegal reference -> abort
   - **not-in-memory** -> continue
2. 빈 frame을 준비
   - 없을 경우 swap
3. 필요한 page를 frame으로 load
   - process의 state는 load 작업이 끝날 때 까지 `wait`
   - Load 완료 후 valid bit을 valid로 변경
   - process의 state를 ready로 바꾸고 ready queue로 옮김
4. CPU가 할당되면 page fault trap이 종료됨
5. Page fault를 일으킨 instruction을 재실행

### Demand Paging

- Pure Demand Paging
  - Program이 처음 시작될 때 부터 아무런 page도 load하지 않는다.
  - 처음 시작될 때 부터 요구되는 사항만을 load
- Locality
  - 특정 기간에는 Page reference들은 매우 작은 부분의 page로부터 발생

### Page Replacement

목표: minimum number of page faults

Page replacement는 reference string을 이용하여 algorithm 별 page replacement 발생 횟수의 차이를 구하여 비교

- FIFO
  - **Belady's Anomaly**: 더 많은 Frame, 더 많은 page fault
- Optimal
  - 최대한 가까운 시일 내로 다시 사용될 page는 victim으로 사용하지 않음
  - 이상적이나, 비현실적
- **LRU, Least Recently Used**
  - **가장 사용된 지 오래된 page가 앞으로도 사용되지 않을 확률이 높다!**
  - Memory 기반의 buffer를 사용하는 모든 곳에서 LRU 기반의 알고리즘을 사용
  - Overhead 두 가지
    - Timestamp needed - extra memory를 page table의 column으로 추가
    - 최소값인 timestamp를 찾는 cost - O(N)
  - OS는 매우 빠른 속도로 작동해야 하므로, approximation model들이 사용됨

### LRU Approximation Algorithms

- Reference bit
  - 초기값이 0인, referenced 횟수를 나타내는 bit를 설정
  - 주기적으로 0으로 reset
  - 어느것이 더 최근인지 구분 불가
- Additional-Reference-Bits Algorithm
  - 여러 reference bit을 사용
  - 시간에 따라 bit을 이동시켜 history를 유지
  - Overhead 두 가지
    - 추가적인 bit 공간
    - 여전히 reference bit을 탐색해야 하는 cost - O(N)
- Second Chance (clock) Algorithm
  - LRU와 유사하며, 작은 overhead
  - 처음 호출 시 1의 초기값을 가진 reference bit 사용
  - Circular queue of pages
    - Victim(`pop`)을 가리키는 pointer가 존재
  - Victim pointer가 가리키며 swap 시기가 될 때 마다 reference bit = 0
  - 0인 상태에서 page fault 발생 시 swap-out
  - 다시 사용될 때 마다 reference bit = 1
  - 최악의 경우 모든 bit = 1일 때 FIFO와 동일

### Thrashing

- Frame 개수가 충분하지 않을 때, page fault가 매우 빈번하게 발생하는 가운데, 다음 세 가지가 악순환으로 발생
  - 이로 인해 CPU utilization이 매우 낮아짐
  - OS는 CPU가 충분히 일을 하지 않고 있다고 판단, process를 추가 할당
  - 따라서 메모리 공간은 더욱 부족해짐

이런 악순환의 현상을 thrashing이라고 한다. Thrashing을 막기 위해서는 **locality**를 잘 이해해야 한다.

### Locality

- Program의 메모리 참조는 고도의 지역성을 가짐

- 임의 시간 $\triangle t$ 내에는 프로그램의 일부분만을 집중적으로 참조

  - 시간 지역성(Temporal Locality)
    - 현재 참조된 메모리가 가까운 미래에도 참조될 가능성이 높음
  - 공간 지역성(Spatial Locality)
    - 하나의 메모리가 참조되면, 주변의 메모리가 계속 참조될 가능성이 높음

  