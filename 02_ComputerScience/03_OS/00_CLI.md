# Command Line Interface

---

- Unix shell(Linux, Git Bash, ...)

- CMD, Powershell

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

- `~/.bahs_profile` 에 적힌 내용들을 유효화 하려면 `source ~/.bash_profile` 을 터미널에 작성해야 하는데,

  이를 자동화 하기 위해 `.zshrc`에 내용 추가

  ```shell
  source ~/.bash_profile
  ```

---

## CMD

- `드래그 + 마우스우클릭` : 복사

- `start .` : 현재 디렉토리 열기

  `explorer . `

  `start explorer . `