# Command Line Interface

---

: 유닉스 shell(Linux, Git Bash, ...)

: CMD, Powershell

---

## PATH

- Shell이 설정된 Path에 해당 프로그램의 존재 여부 확인

  존재 시, 어느 위치에서도 해당 프로그램을 사용할 수 있도록 도움

- Windows의 경우 User보다 System 계정의 Path를 먼저 고려(Override)

  Linux의 경우 import된 순서로 Override

- Linux의 경우 `sudo` 입력을 통해 관리자 Override가 가능하며,

  Windows는 '관리자 권한으로 실행'을 사용
  
- terminal에서 jupyter notebook 명령어 단축시키기

  ~/user/students 위치에서 

  - `code ~/.bashrc` 
  - `alias jn='jupyter notebook'` 기입
  - terminal > `source ~/.bashrc`

  이후 terminal에서 jn만 작성해도 jupyter notebook 실행

---

## Unix Shell

`sudo + 명령어` : 관리자 권한으로 명령어 실행

`ls` : 현재 디렉토리 목록

`ls -al` : 숨김파일까지 목록

`cd`: 지정 위치로 이동

`pwd`: 현위치 (point working directory)

`mkdir`: 폴더 생성

`code .`: vs code 현위치에서 실행

`touch 문서.txt`: 문서를 생성

`rm 문서.txt`: 문서 삭제

`rm -rf directory/file` : 묻지도 따지지도 않고 해당 directory/file 강제 제거

`mv 현재명 변경명` : 파일명 변경, 위치 이동

`.` : 현위치

`./` : 현위치에서 부터 path 설명

`~/` : home으로부터 path 설명

`chmod` : 권한 부여

tar

`echo` : cli의 print

`echo $PATH` : path 보여줌

---

## CMD

`드래그 + 마우스우클릭` : 복사

