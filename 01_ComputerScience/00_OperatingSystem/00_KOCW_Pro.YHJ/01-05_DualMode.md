# 1-5. Dual Mode

대부분의 운영체제는 이중 모드를 지원하며, OS가 이중 모드를 통해 보호하고자 하는 대상은 다음 세가지이다.

1. 입출력 장치
2. 메모리
3. CPU

<br><br>

## 이중 모드, Dual Mode

서버 컴퓨터는 여러 사람에게 동시에 사용될 수 있는 환경이다. 개인 PC일 지라도 여러 개의 프로그램을 동시에 사용한다. 여러 사용자 혹은 프로그램들이 STOP, HALT, RESET과 같은 명령어를 남용한다면 큰 문제가 발생할 수 있다.

따라서 대부분의 운영체제(스마트폰 포함)는 이중 모드를 지원한다. 이중 모드는 관리자(Supervisor) 모드와 사용자(User) 모드로 구성된다.

- 관리자(=시스템, 모니터, 특권) 모드

  Superviser(=System, Monitor, Previliged) Mode

- 특권 명령, Privileged Instructions

  STOP, HALT, RESET, SET_TIMER, SET_HW, ...

- 권한이 없는 사용자가 위 명령을 시도하면 `Privileged Instructions Violation` 발생

이중 모드는 레지스터(Register)의 플래그(Flag)를 사용하여 조작한다.

<br>

### CPU에서 이중 모드 제어하기

- CPU의 구조
  - 레지스터, Register
  - 산술 논리 장치, ALU(Arithmetic Logic Unit)
  - 제어 유닛, Control Unit
- CPU Register의 다섯 가지 플래그(Flag)
  - `Carry`: 자리수 상승을 표현
  - `Negative`: 계산 결과 값이 음수
  - `Zero`: 계산 결과 값이 0
  - `Overflow`
  - `Dual Mode`
    - `1`: 관리자 모드
    - `0`: 사용자 모드

마우스, 키보드와 같은 I/O 장비 또한 관리자 모드로만 조작할 수 있으며, 사용자 프로그램 실행 시 사용자 모드로 flag가 변환된다. 사용자 프로그램 사용 중에도 마우스 및 키보드 조작, H/W에 접근, 프린터 및 모니터 사용이 필요하다면, OS에 요청하여 관리자 모드로 전환하고, 해당하는 SW Interrupt를 발생시켜 Interrupt Service Routine을 발생시켜야 한다. 만약 사용자 프로그램이 직접 ISR을 조작하게 한다면, 보안 등에서 큰 문제가 발생할 수 있다.



