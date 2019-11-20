# Git

Git의 기능 두가지

1. SCM, Source Code Manager

2. VCS, Version Control System

3. 원격 저장소, Remote Repository

   Github, Gitlab, Bitbucket

4. 협업 도구

   `Push + Pull` : 차례로 `Push`와 `Pull`을 진행

   `Fork + PR(Pull Request)` : 관리자의 Repository를 `Fork`해와서 수정 후 `Pull Request`

   `Branch + PR` : 각자 작업하는 Branch를 Push하여 Github에서 PR하여 `Push`

<br><br>

## Intro

1. Status

   어떤 file을 `add`하여 Staging Area에 update 할 것인지 결정 후 `add`

2. Staging Area

   - `add` 된 file들을 remote repository에 update 할 것인지 결정 후 `commit`

   - **`commit` 한 내용들은 `git log`에서 확인 가능**하며,

   - 언제든 `git checkout 'commit hash'` 혹은 `git reset 'commit hash'`를 이용하여 해당 시점의 local state로 변경 가능

   - `commit` 은 **logical한 개발 단위**이므로, 중간 중간에 `commit`을 진행(ex. 기능 단위)

3. Commit Log

   staging area에서 `commit`한 내용들을 누적하여 기록

   - **HEAD**

     현재 Working Directory의 위치를 가리킴

<br>

### Git Env, Language

다음 내용을 `~/.bash_profile`에 추가합니다.

```bash
alias git='LANG=en_GB git'
```

<br>

### Config

```shell
# git 정보 확인
$ git config --list

# git에 유저 정보 기록
$ git config --global user.email "cdb921226@gmail.com"
$ git config --global user.name "jarvis08"

# 로그인 정보 잘못 입력했을 때
$ git credential reject
$ protocol=https
$ host=github.com
```

<br>

### Remote Repository

```shell
# Making new repository
$ git init
# .git이라는 하위 디렉토리 생성
# .git에는 저장소에 필요한 뼈대 파일(skeleton)을 생성

# 삭제
$ rm -r .git

# 현재 연결된 git 정보 확인(이름)
$ git remote
# 현재 연결된 git 정보 확인(이름, 주소)
$ git remote -v

# remote repository(원격저장소) 추가
$ git remote add 설정할저장소이름 저장소주소

# Cloning
$ git clone address/repository.git
# 원하는 디렉토리명으로 repository를 clone
$ git clone address/repository.git new_name
```

<br>

### Status & Log

```shell
# git 내용 정리
$ git status
# git commit 이력 검색
$ git log

# 보기 어려우므로 commit들을 한 줄로 요약하여 표기, sha1 16진수 해쉬값은 7글자로.
$ git log --oneline

# 그림으로 확인 가능
$ git log --oneline --graph

# 아래는 develop branch와 master branch를 합친 예시
$ git log --oneline --graph
*   728f351 (HEAD -> master) Merge branch 'develop'
|\
| * ad81ddd (develop) schenario
* | 217ffe6 html made
|/
* efb7d6f merge
* 0265663 merge fast-forward merge scenario

```

<br>

### Add & Commit

```shell
# Staging Area로 data를 전달
$ git add file_name
$ git add .

# add 취소, unstage
$ git rm --cached 파일명(경로포함)
$ git reset HEAD 파일명

# add 전체 취소
$ git reset HEAD

# Staging Area - commit
# -m : with message
$ git commit -m "commit message"
# add되지 않은 것 까지, git status 상에 나오는 변화된 모든 내용을 commit
$ git commit -a
```

- 커밋은 Git 저장소에 여러분의 디렉토리에 있는 모든 파일에 대한 스냅샷을 기록하는 것

- 디렉토리 전체를 복사하여 붙여넣는것과 유사하지만, 훨씬 유용한 방법

- Git은 가능한 한 커밋을 가볍게 유지하고자 하며,

  커밋할 때마다 디렉토리 전체를 복사하진 않는다.

- 각 커밋은 저장소의 이전 버전과 다음 버전의 변경내역("delta"라고도 함)을 저장

- 그래서 대부분의 커밋이 그 커밋 위의 부모 커밋을 가리킴

- 저장소를 복제(clone)하려면 모든 변경분(delta)를 풀어내야 하는데,

  이 때문에 명령행 결과로 아래 문구를 확인 가능

  `resolving deltas`

<br>

### Checkout

```shell
# 10단계 이전의 commit으로 이동
$ git checkout HEAD~10
# 특정 시점으로 이동
$ git checkout 'log의 commit hash code 6자리'
# 복귀
$ git checkout master
```

<br>

### HEAD

먼저"HEAD"에 대해 이야기해 봅시다. HEAD는 현재 체크아웃된 커밋을 가리킵니다. -- 다시 말하자면 현재 작업중인 커밋입니다.

HEAD는 항상 작업트리의 가장 최근 커밋을 가리킵니다. 작업트리에 변화를 주는 git 명령어들은 대부분 HEAD를 변경하는것으로 시작합니다.

일반적으로 HEAD는 브랜치의 이름을 가리키고있습니다(bugFix와 같이). 커밋을 하게 되면, bugFix의 상태가 바뀌고 이 변경은 HEAD를 통해서 확인이 가능합니다.

<br>

### 상대 참조

말했듯이, 커밋들을 해시로 구분하고 사용하는것이 아주 편하다고 볼 수는 없습니다. Git의 상대 참조(Relative Ref)가 여기서 등장합니다. 굉장한 기능입니다.

상대 참조로 우리가 기억할 만한 지점(브랜치 `bugFix`라던가 `HEAD`라던가)에서 출발해서 이동하여 다른 지점에 도달해 작업을 할 수 있습니다.

상대 커밋은 강력한 기능인데, 여기서 두가지 간단한 방법을 소개하겠습니다.

- 한번에 한 커밋 위로 움직이는 `^`

  ex) `git checkout HEAD^`

- 한번에 여러 커밋 위로 올라가는 `~<num>`

<br>

### Push & Pull

```shell
# remote repository에 적용하기
$ git push -u origin master
# -u origin master : 첫 push 이후 생략 가능
$ git push

# remote의 파일 local로 복사
$ git pull
```

`pull`은 `fetch`와 `merge`가 합쳐진 기능이며, `pull`로 인해 conflict 발생 가능

<br>

### cached

`git add` 취소하기

```shell
$git rm --cached -r fileName
$git rm --cached -r .
```

<br>

### Reset

Git에게 치명적인 역할을 수행하는 명령어이며, 사용시 매우 주의

```shell
$ git reset
# 해당 시점으로 완전 변경하며, 해당 commit 시점 이후의 모든 기록을 삭제(복원 불가)
$ git reset --hard 'log의 commit hash code'
$ git reverse

# [방법 1] commit을 취소하고 해당 파일들은 staged 상태로 워킹 디렉터리에 보존
$ git reset --soft HEAD^
# [방법 2] commit을 취소하고 해당 파일들은 unstaged 상태로 워킹 디렉터리에 보존
$ git reset --mixed HEAD^ // 기본 옵션
$ git reset HEAD^ // 위와 동일
$ git reset HEAD~2 // 마지막 2개의 commit을 취소
# [방법 3] commit을 취소하고 해당 파일들은 unstaged 상태로 워킹 디렉터리에서 삭제
$ git reset --hard HEAD^

### commit 이후 message 변경
# [방법 1] commit을 취소하고 해당 파일들은 staged 상태로 워킹 디렉터리에 보존
$ git reset --soft HEAD^
# [방법 2] commit을 취소하고 해당 파일들은 unstaged 상태로 워킹 디렉터리에 보존
$ git reset --mixed HEAD^ // 기본 옵션
$ git reset HEAD^ // 위와 동일
$ git reset HEAD~2 // 마지막 2개의 commit을 취소
# [방법 3] commit을 취소하고 해당 파일들은 unstaged 상태로 워킹 디렉터리에서 삭제
$ git reset --hard HEAD^
```

- TIP git reset 명령은 아래의 옵션과 관련해서 주의하여 사용해야 한다.

  - reset 옵션
    –soft : index 보존(add한 상태, staged 상태), 워킹 디렉터리의 파일 보존. 즉 모두 보존.
    –mixed : index 취소(add하기 전 상태, unstaged 상태), 워킹 디렉터리의 파일 보존 (기본 옵션)
    –hard : index 취소(add하기 전 상태, unstaged 상태), 워킹 디렉터리의 파일 삭제. 즉 모두 취소.

- TIP 만약 워킹 디렉터리를 원격 저장소의 마지막 commit 상태로 되돌리고 싶으면, 아래의 명령어를 사용한다.
  단, 이 명령을 사용하면 원격 저장소에 있는 마지막 commit 이후의 워킹 디렉터리와 add했던 파일들이 모두 사라지므로 주의해야 한다.

  ```shell
  # 워킹 디렉터리를 원격 저장소의 마지막 commit 상태로 되돌린다.
  $ git reset --hard HEAD
  ```

  https://gmlwjd9405.github.io/2018/05/25/git-add-cancle.html

- Rebasing

  실수로 빠뜨린 어떤 내용을 같이 커밋하고 싶다 할 때 사용

  commit을 다시 하여 합침

<br>

### Remote Repository 재할당

```shell
# 이미 origin이 있기 때문에 gitlab을 다시 origin으로 설정해주는 작업을 시작
$ git remote add origin https://lab.ssafy.com/mtice/cho_dong_bin.git
fatal: remote origin already exists.


$ git remote -v
origin  https://github.com/jarvis08/hw_ws.git (fetch)
origin  https://github.com/jarvis08/hw_ws.git (push)

$ git remote remove origin
$ git remote add origin https://lab.ssafy.com/mtice/cho_dong_bin.git

$ git remote -v
origin  https://lab.ssafy.com/mtice/cho_dong_bin.git (fetch)
origin  https://lab.ssafy.com/mtice/cho_dong_bin.git (push)

$ git push -u origin master

$ git remote add github https://github.com/jarvis08/hw_ws.git
$ git remote -v
github  https://github.com/jarvis08/hw_ws.git (fetch)
github  https://github.com/jarvis08/hw_ws.git (push)
origin  https://lab.ssafy.com/mtice/cho_dong_bin.git (fetch)
origin  https://lab.ssafy.com/mtice/cho_dong_bin.git (push)

# origin을 gitlab으로 교체했기 때문에 origin master는 gitlab
$ git push origin master
# github이라는 이름으로 추가했기 때문에 github master로 push
$ git push github master

```

<br>

### 이전 Commit으로 이동 후, 다시 이전 Commit 상태로 되돌리기

```shell
$ git log --oneline
# 571b4d1 (HEAD -> master) add remote related commmands
# 554ccf3 first commit

$ git checkout 554ccf3
# HEAD가 554ccf3 커밋 상태로 옮겨감
# 당시 커밋 상태로 파일 내용이 변경됨

$ git checkout master
# HEAD를 다시 master라는 branch인 571b4d1로 옮겨옴
# 우리는 알게 모르게 master라는 branch에서 작업하고 있었던 것이다!
```







