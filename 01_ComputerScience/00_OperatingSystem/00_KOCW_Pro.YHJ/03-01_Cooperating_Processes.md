# 3. Process Synchronization

# 3-1. Cooperating Processes

프로세스는 크게 **Independent Processd**와 **Cooperating** Process로 나눌 수 있습니다. 그 중 Cooperating Porcess는 **시스템 내의 다른 프로세스로부터 영향을 주거나 받는 프로세스**입니다. Cooperating Process의 예시는 다음이 있습니다.

- 프로세스 간 통신

  전자 우편, 파일 전송

- 프로세스 간 자원 공유

  메모리 상의 자료들, 데이터 베이스(수강신청, 주식거래, 콘서트 예매) 등

<br>

<br>

## 3-1-1. 프로세스 동기화, Process Synchronization

프로세스 간 공유되는 데이터(**공통 변수, Common Variable**)로의 **Concurrent Update**는 데이터의 **일관성(consistancy)**을 파괴할 수 있습니다. Cooperating 프로세스들의 순서를 정렬하여 실행하면 데이터의 일관성을 유지할 수 있습니다. 따라서 **임계 구역**에 대해 한 번에 한 쓰레드만 업데이트 할 수 있도록 조치해야 하며, 우리는 이러한 작업을 **프로세스 동기화(Porcess Synchronization**)라고 합니다.

_임계 구역에 대한 정의와 자세한 내용은 3-2에서 알아보겠습니다._

<br>

### 프로세스 동기화 미고려 시 발생하는 문제

부모와 자식이 하나의 용돈 계좌를 공유한다고 상황을 설정해 봅시다. 부모는 용돈 계좌에 용돈을 10000원 씩 입금하며, 자식은 10000원 씩 출금합니다. 대략적인 내용은 다음과 같으며, 

```java
class BackAccount {
		int balance;
	  void deposit(){
        balance = balance + 10000
    }
		void withdraw(){
				balance = balance - 10000
    }
}
				
b = new BankAccount();
p = new Parent(b);
c = new Child(c);
// Parent와 Child는 각각의 Thread로 실행되며,
// Class 구현 및 run(), start(), join()하는 과정은 생략
p.deposit();
c.withdraw();
```

그런데 계좌의 출금과 입금을 각각의 쓰레드로 하여 동일 회수(i.g., 1000회)만큼 반복하여 실행한다면, 결과값은 0이 아닌 다른 값이 됩니다. 결과값이 달라지는 이유는 `balance = balance +- 10000` 에 있습니다.

Java, C, Python과 같은 **High Level Language**들은 한 줄의 식으로 입출금 내용을 계산할 수 있습니다. 하지만 실질적으로 Registry를 조작하여 데이터를 업데이트 하는 **Low Level Laguage**인 Assembly의 경우 다음과 같은 과정을 거쳐야 합니다.

```assembly
ldr r0, [balance]
ldr r1, [amount]
add r0, r0, r1
str r0, [balance]
```

위 과정 중간에 Switching이 발생하여 update 도중에 Concurrent Update가 발생한다면, register의 값이 달라질 수 밖에 없습니다. 따라서 이를 해결하기 위해서는 쓰레드를 **Atomic**하게 한 덩어리로 실행해야 임계 구역에서 문제가 발생하지 않습니다.