# 4-2. 메모리 주소, Address of Memory

CPU는 주소(Address)를 Main Memory에 전송하고, Main Memory는 해당 주소에 존재하는 Data를 CPU에게 반환합니다.

<br>

<br>

##4-2-1. 프로그램 개발

### 원천 파일, Source File

**High Level Language**(i.g., C, C++, Java), 혹은 **Assembly Language**

<br>

### 목적 파일, Object File

**Compiler**에 의한 **Compile**, 혹은 **Assembler**에 의한 **Assemble**의 결과물

<br>

### 실행 파일, Executable file

**Linker**에 의한 **Link**의 결과물

**Loader**가 **Executable File**을 Main Memory에 **Load**합니다.

![Program_Development](./assets/Program_Development.png)<br>

<br>

## 4-2-2. 프로그램 실행

프로세스의 메모리 = `Code` + `Data` + `Stack`

- `Stack`
  1. 함수 호출 시 돌아와야 할 주소를 저장
  2. Local Variable wjwkd

OS는 다음과 같은 사용자의 역할을 대신해 준다.

- 메모리의 어느 주소에 프로세스를 Load 할 것인가?
- 다중 프로그래밍 환경에서는 어떻게 할 것인가?

<br>

<br>

## 4-2-3. MMU 사용

MMU에는 총 세 가지 레지스터가 존재합니다. 1~2번은 이전에 설명한 내용이니 생략하겠습니다.

<br>

### 1. Base Register

<br>

### 2. Limit Register

<br>

### 3. Relocation Register

i.g., CPU가 0번 주소의 데이터를 요청했다. 하지만 Relocation Register가 주소를 조작하여 500번 주소의 Data를 CPU에게 반환한다. 그리고 CPU는 그 사실을 알지 못한다.

![MMU](./assets/MMU.png)

<br>

<br>

## 4-2-4. 주소 구분

주소를 **논리 주소(Logical Address)**와 **물리 주소(Physical Address)**로 구분합니다.

