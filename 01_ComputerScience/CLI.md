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
  
- Terminal 명령어 단축시키기

  1. Home Directory 이동

     - bash file
       - Linux/Windows 
  
       `~/.bashrc`
  
       - OS X
  
         `~/.bash_profile`
  
  2. 에디터로 bash file 열기
  
     `code ~/.bashrc` 
  
     `vi ~/.bashrc`
  
  3. `alias jn='jupyter notebook'` 기입
  
  4. Terminal에서 수정 사항을 적용  `source ~/.bashrc`
  
     이후 Terminal에서 `jn`만 작성해도 jupyter notebook 실행
  
- 절대경로 / 상대경로

  - 절대 경로

    - Windows
      - root 경로 = `C:\` 
    - Linux
      - root 경로 = `/`
    - `~/` : home directory

  - 상대 경로

    root 경로 = 자기 자신(working directory)

    - `../../directory/`

      두 단계 거슬러 올라간 후, direc 이라는 directory

---

## Unix Shell

- `sudo + 명령어` : 관리자 권한으로 명령어 실행

  `chmod` : 권한 부여

  `.` : 현위치를 의미하는 도구

  `./` : 현위치에서 부터 path 설명

  `~/` : home으로부터 path 설명

- `ls` : 현재 디렉토리 목록

  `ls -al` : 숨김파일까지 목록

- `cd`: 지정 위치로 이동

- `pwd`: 현위치 (point working directory)

- `mkdir`: 폴더 생성

- `touch 문서.txt`: 문서를 생성

- `rm 문서.txt`: 문서 삭제

  `rm -rf directory/file` : 묻지도 따지지도 않고 해당 directory/file 강제 제거

- `mv 현재명 변경명` : 파일명 변경, 위치 이동

- `echo` : CLI의 `print()`

  `echo $PATH` : path 보여줌

- `code .`: vs code 현위치에서 실행

---

## OS X

- `open . ` : 현재 directory를 finder에서 열기
- `Keka`
  - `Ctrl + Shift + C : Compress`
  - `Ctrl + Shift + X` : Extract
  - `Ctrl + Shift + K` : Send to Keka

- Customized Key

  `Ctrl + Option + Super` : 현재 finder directory에서 iterm2 열기

---

## CMD

`드래그 + 마우스우클릭` : 복사