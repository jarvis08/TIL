# 3-4. Synchronization Problems

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

`empty.value = buffer_size`

`full.value = 0`

<br>

<br>

## 3-4-2. Readers-Writers Problem

공통 데이터베이스를 공유하며 발생하는 문제이며, 데이터베이스 자체가 Critical Section입니다. 문제에서는 사용자를 두 부류로 나눕니다.

- Readers
  - read data
  - never modify data
- Writers
  - read data
  - modify data

사용자들이 데이터베이스를 사용하는 조건에는 다음 세 가지가 있습니다.

1. Reader들 끼리는 동시에 data를 읽을 수 있다.
2. Writer들 끼리는 Mutual Exclusion이 적용되어, 동시에 데이터를 조작할 수 없다.
3. Writer와 Reader는 동시에 데이터를 읽거나 조작할 수 없다.

문제의 규칙은 다음 세 가지 중 하나를 추가적으로 선택하여 정의됩니다.

- The first R/W Problem (Readers Preference)
- The second R/W Problem (Writers Preference)
- The third R/W Problem (No Preference)

<br>

<br>

## 3-4-3. Dining Philosopher Problem

이 문제의 내용은 다음과 같습니다.

- 5 명의 철학자가 5 개(쌍X)의 젓가락을 공유
- 철학자가 식사를 할 때에는 젓가락 2개를 사용해야 한다.
- 철학자는 (생각 > 식사 > 생각 > 식사 > ...)와 같이 생각과 식사를 번갈아가며 진행
- 철학자들은 원탁에 둘러앉아 있으며, 철학자들 사이 마다 젓가락이 1 개씩 놓여져 있다.
- 철학자들은 각자 젓가락을 잡을 때 자신의 왼쪽 젓가락을 잡은 후 오른쪽 젓가락을 잡는다.

문제를 풀기 위해 해석하자면, 젓가락들은 Critical Section이므로 `value`가 1인 세마포로 사용되어야 합니다.

```java
lstick.acquire()
rstick.acquire()
eating()
lstick.release()
rstick.release()
tihnking()
```

<br>

### Starvation

하지만 코드 실행 시 처음에는 번갈아가며 잘 식사를 하는듯 싶더니, 중간 부터는 모두가 식사를 할 수 없게 됩니다. 그 이유는 철학자들이 **Starvation** 상황을 맞닥뜨렸으며, 그 원인으로 **교착 상태**(**Deadlock**)에 빠졌기 때문입니다. 모든 철학자들이 자신의 왼쪽 젓가락을 `acquire`한 상태이며, 오른쪽 젓가락(오른쪽 사람에게는 왼쪽 젓가락, 즉 오른쪽 사람이 사용 중인 젓가락)을 `acquire` 하고자 기다리고 있습니다.

