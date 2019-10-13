# 3-4 Synchronization Problems

전통적인 동기화 문제들(Classical Synchronization Problems)은 다음과 같습니다.

1. Producer and Consumer Problem

   생산자 - 소비자 문제

   **유한 버퍼 문제(Bounded Buffer Problem)**를 다룹니다.

2. Readers-Writers Problem

   공유 데이터베이스 접근 문제

   i.g., 티켓 예매

3. Dining Philosopher Problem

   식사하는 철학자 문제

   비현실적인 주제

<br>

<br>

## 3-4-1. Producer and Consumer Problem

생산자가 데이터를 생산하면, 소비자는 생산된 데이터를 소비합니다. 현실 세계의 많은 것들이 이러한 형태로 이루어져 있습니다.

- **High Level Language** > *Compiler* > **Assembly Code** > *Assembler* > **Executable Code**(**Object Code**)
- File Server > Client
- Web Server > Web Client

<br>

### 유한 버퍼, Bounded Buffer

생산된 데이터는 버퍼에 저장되며, 버퍼에 저장되어 있는 데이터를 소비자가 사용합니다.

**Producer** > **Buffer** > **Consumer**

현실 세계에서 버퍼의 크기는 유한합니다. 따라서 생산자는 버퍼가 가득 차면 더 이상 넣을 수 없으며, 소비자는 버퍼가 비었을 때 데이터를 뺄 수 없습니다.

생산자 소비자 문제의 버퍼는 순환 큐(Circular Queue)를 사용하여 구현하는데, 순환 큐에 새로 삽입될 데이터의 인덱스를 가리키는 `in`, 다음으로 빠져나갈 데이터의 인덱스를 가리키는 `out` 변수가 있습니다.

```java
in = (in + 1) % buffer_size
out = (out + 1) % buffer_size
```

buffer_size를 나눔으로서 버퍼가 가득 찼을 때 다시 0 인덱스로 지정되도록 합니다.

<br>

### 상호 배타의 적용, Application of Mutual Exclusion

3-3의 **세마포(Semaphore)** 또한 사용되어야 합니다. 데이터를 버퍼에 넣고 빼는 작업이 이루어질 경우, 버퍼의 데이터 개수를 세는 `count` 변수에 옳지 않은 값이 저장될 수 있습니다. 따라서 **임계 구역(Critical Section)**인 버퍼에 세마포를 적용하여 **상호 배타(Mutual Exclusion)** 조건을 만족시켜야 합니다.

`mutex.value = 1`

<br>

### Busy-wait

만약 버퍼의 크기가 `count`와 같을 경우, CPU는 계속해서 `count`가 감소하길 기다리며 무한 루프에 빠집니다. 이러한 낭비를 busy-wait라고 하며, **세마포**를 활용하여 이를 방지할 수 있습니다. 그리고는 **소비자가 버퍼에서 데이터를 빼 낼 때**, 세마포에서 생산자를 다시 release 합니다.

`empty` 세마포는 버퍼가 비어있는 상황을 감시하며, **소비자로 하여금 버퍼가 비었을 때 busy-wait 하지 않도록** 하기 위해 사용합니다. `full` 세마포는 가득 차는 것을 감시하며, **생산자로 하여금 버퍼가 가득 찼을 때 busy-wait 하지 않도록** 하기 위해 사용합니다.

empty.value = buffer_size`

`full.value = 0`

