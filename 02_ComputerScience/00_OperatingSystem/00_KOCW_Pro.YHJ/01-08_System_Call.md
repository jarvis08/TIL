# 1-8. 시스템 콜, System Call

## 1-8-1. 시스템 콜의 정의

Application이 운영체제 서비스(OS Service)를 받기 위해 호출하는 행위

<br>

<br>

## 1-8-2. 주요 시스템 콜

### Process

- end

  프로세스를 종료

- abort

  프로세스를 강제 종료

- load

  H/W 프로그램을 메모리로 로드

- execute

- create

- terminate

- get/set attributes

  프로세스의 속성(ID, 메모리 사용량 등)을 읽음/설정

- wait events

- signal event

<br>

### Memory

- allocate
- free

<br>

### File

- create, delete
- open, close
- read, write
- get/set attributes

<br>

### Device

- request
- release
- read, write
- get/set attributes
- attach/detach devices

<br>

### Information

- get/set time
- get/set system data

<br>

### Communication

- socket
- send, receive

<br>

<br>

## 1-8-3. 시스템 콜 만들어 보기

파일을 생성하는 과정으로 예를 들어 보겠습니다.

1. Assembly 언어로 파일 속성을 정의
2. OS System Call 요청

코드들은 각 운영체제의 System Call Library에서 제공하는 형태로 작성합니다.

<br>

### MS-DOS

```shell
AH = 3CH
CX = file_attributes
DS:DX = file_name
INT = 80H
```

- `INT`: Interrupt

<br>

### Linux

```shell
# 여기서의 8은 파일을 생성할 때 사용
EAX = 8
ECX = file_attributes
EBX = file_name
INT = 80H
```





