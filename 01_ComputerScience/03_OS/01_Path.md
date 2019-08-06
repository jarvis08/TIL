# PATH

---

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

       - 에디터로 bash file 열기

         `code ~/.bashrc` 

         `vi ~/.bashrc`

       - `alias jn='jupyter notebook'` 기입

       - Terminal에서 수정 사항을 적용  `source ~/.bashrc`

         이후 Terminal에서 `jn`만 작성해도 jupyter notebook 실행

---

## 절대경로 / 상대경로

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