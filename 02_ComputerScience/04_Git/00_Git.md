# Git & Github

---

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

   ```shell
   # git 정보 확인
   $ git config --list
   
   # github 로그인
   $ git config --global user.email "cdb921226@gmail.com"
   $ git config --global user.name "jarvis08"
   
   # 로그인 정보 잘못 입력했을 때
   $ git credential reject
   $ protocol=https
   $ host=github.com
   
   # git 내용 정리
   $ git status
   # git commit 이력 검색
   $ git log
   
   # Making new repository
   $ git init
   # .git이라는 하위 디렉토리 생성
   # .git에는 저장소에 필요한 뼈대 파일(skeleton)을 생성
   
   # 현재 연결된 git 확인
   $ git remote -v
   
   # Cloning
   $ git clone address/repository.git
   # 원하는 디렉토리명으로 repository를 clone
   $ git clone address/repository.git new_name
   
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
   
   # remote repository에 적용하기
   $ git push -u origin master
   # -u origin master : 첫 push 이후 생략 가능
   $ git push
   
   # remote의 파일 local로 복사
   $ git pull
   
   $ git rm -r 파일/디렉토리명
   $ git mv 현재이름 바꿀이름
   
   # 10단계 이전의 commit으로 이동
   $ git checkout HEAD~10
   # 특정 시점으로 이동
   $ git checkout 'log의 commit hash code 6자리'
   # 복귀
   $ git checkout master
   
   $ git reset
   # 해당 시점으로 완전 변경하며, 해당 commit 시점 이후의 모든 기록을 삭제(복원 불가)
   $ git reset --hard 'log의 commit hash code'
   $ git reverse
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

------

- 기존의 origin master를 설정 해제한 후,

  새로운 remote repository 할당하기
  
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
