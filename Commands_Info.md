# Commands

---

## Git

`add` `commit` 의 경우 local에서 작동하며,
`push`를 통해 remote로 밀어 넣음

```shell
$ git config --global user.email "cdb921226@gmail.com"
$ git config --global user.name "jarvis08"

### Making new repository
$ cd (master directory)
$ git init
# 기존 프로젝트를 git으로 관리하고 싶을 때, 프로젝트의 디렉토리로 이동해서 명령
# .git이라는 하위 디렉토리 생성
# .git에는 저장소에 필요한 뼈대 파일(skeleton)을 생성
$ git add file_name
$ git add .
$ git commit -m "commit message"
# -m : with message
$ git commit -a
# add되지 않은 것 까지, git status 상에 나오는 변화된 모든 내용을 commit
$ git checkout -- <file>...
# to discard changes in working directory
$ git remote add origin https://github.com/jarvis08/SSAFY.git

### Push & Pull
$ git push -u origin master
# -u origin master : 첫 push 이후 생략 가능
$ git push
$ git pull
# remote의 파일 local로 복사

### Cloning
$ git clone address/repository.git
$ git clone address/repository.git new_name
# 원하는 디렉토리명으로 repository를 clone해옴

$ git status
$ git log
$ git rm -r 파일/디렉토리명
$ git mv 현재이름 바꿀이름
$ git merge
$ git checkout 시점
# 이전 버전으로 이동 (git log 활용)
$ git reset
$ git reverse
```

---

## CLI(Command Line Interface)

: 유닉스 shell(Linux, Git Bash, ...)
: CMD, Powershell

> `ls` : list 목록
> `cd`: 지정 위치로 이동
> `pwd`: 현위치 (point working directory)
> `mkdir`: 폴더 생성
> `code .`: vs code 현위치에서 실행
> `touch 문서.txt`: 문서를 생성
> `rm 문서.txt`: 문서 삭제

---

## Markdown Language

> `- + Enter`: black point
> `(```python + Enter)`: 블록 코드 강조 (back tick)
>
> ```python
> (Ctrl + Enter) : 빠져나가기
> ```
>
> `--- + Enter`: 줄긋기
> `Ctrl + T`: 표 만들기
> `back tick + 내용 + back tick`: 코드블록에 내용 넣기
> `> ` 단 띄우기

---

## MOOC Lectures

: Massive Online Open Courses
ex) Coursera, edX, Udacity(Stanford University), edwith

- **Git**
  생활코딩 - 지옥에서 온 git
  Udacity - 무료강의 중 Git
- **Computer Science**
  edX, edwith(한글번역) - CS50
- **Machine Learning**
  Udacity - ML