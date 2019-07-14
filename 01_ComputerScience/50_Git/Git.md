# Git

---

`add` `commit` 의 경우 local에서 작동하며,

`push`를 통해 remote로 밀어 넣음

`.gitignore` : 해당 파일에 작성해 둔 파일들은 git 활동에서 제외

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