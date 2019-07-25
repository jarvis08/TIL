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
   # github 로그인
   $ git config --global user.email "cdb921226@gmail.com"
   $ git config --global user.name "jarvis08"
   
   # 로그인 정보 잘못 입력했을 때
   git credential reject
   protocol=https
   host=github.com
   
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
   
   # Staging Area - commit
   # -m : with message
   $ git commit -m "commit message"
   # add되지 않은 것 까지, git status 상에 나오는 변화된 모든 내용을 commit
   $ git commit -a
   
   
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

---

## .gitignore

- git에 upload하고 싶지 않은 대상을 `.gitignore`에 작성,

  해당 파일에 작성해 둔 파일들은 git 활동에서 자동으로 제외

  - 자신의 directory를 기준으로, 모든 하위 directory까지 영향

  - `파일명/디렉토리명` : file, directory 제외
  - `디렉토리명/*` : directory 내용물을 모두 제외

  ```shell
  # jupyer notebook checkpoints 제거
  $ code .gitignore
  >> 내부 작성 .ipynb_checkpoints
  
  # 현재 디렉토리의 .DS_Store 제거
  $ git rm --cached .DS_Store
  # 모든 remote git repository에서 .DS_Store 찾아서 삭제
  find . -name .DS_Store -print0 | xargs -0 git rm --ignore-unmatch
  # 앞으로의 모든 repository에서의 .DS_Store upload 예방
  # 1. global하게 사용할 .gitignore를 어딘가에 생성.
  # 파일 예시
  echo .DS_Store >> ~/.gitignore_global
  # 2. git에게 모든 repository에 사용할 것을 선언하기
  git config --global core.excludesfile ~/.gitignore_global
  # 참고자료 - https://stackoverflow.com/questions/18393498/gitignore-all-the-ds-store-files-in-every-folder-and-subfolder
  
  # TIL/50_SSAFY/8ython directory 내부에 .git이 따로 있었으나, 제외하고 다시 TIL의 .git으로 포함시킴
  $ git rm -rf ~/TIL/50_SSAFY/8ython.git
  $ git rm --cached 8ython/
  검색창 :: gitignore.io/api/python
  ```

- Rebasing

  실수로 빠뜨린 어떤 내용을 같이 커밋하고 싶다 할 때 사용

  commit을 다시 하여 합침

---

## Mater / Branch

- master에서 branch를 생성

  master와 branch가 서로 영향을 주지 않기 때문에 병행 개발이 가능

- 현업 예시

  - master = 제품
  - branch = 신기능

  ```shell
  # branch 확인
  $ git branch
  # branch 생성
  $ git branch godbin
  # branch로 checkout
  $ git checkout godbin
  # branch 생성 및 checkout
  $ git checkout -b godbin
  
  # branch add, commit 이후 push 방법
  $ git push origin godbin
  
  # fast-forward merge, 빨리감기 병합
  # HEAD는 master에 위치
  $ git checkout master
  $ git merge godbin
  # 이후 다시 push
  
  # branch 삭제
  $ git branch -d godbin
  ```

## Git Co-Working

- git info 열람

  - `ls .git`
  - `cat .git/config`

- co-working 방법 두 가지

  1. 시간을 분리
  2. 파일을 분리

- Team Leader는 작업을 분배

  ------

## Github 제공 기능

- Github `organization`

  github에서 제공하는 gitlab의 `group` 기능

  - 만약 owner가 아닌 member로 설정되어 있을 경우,

    repository 별로 member의 권한을 설정해줘야함채

- Github `Issues`

  - 게시판 느낌으로 `New issue` 후 댓글 달듯이 Write를 통해 `@member` 설정하여, issue를 꼭 봐야하는 사람을 태그
  - issue 해결 후 `closed`를 통해 종료
  - `reopen` 도 가능

- Work Flow

  git work-flow : 매우 복잡

  gitlab work-flow가 github 보다 조금 더 간략

------

## Handling Conflicts

1. 두 팀원이 동시에 `commit`을 진행

   - 나

     `Readme.md` 수정하여 `push`

   - 팀장

     `pull` 하지 않은 채, `test.py` 파일을 생성하여 `push` 시도

     But, `push`가 이루어지지 않는다!

2. merge 시도, `auto merge`

   - 우선 `pull`을 진행하여 `commit`의 시간 순서로 file을 정렬

   - `HEAD` 포인터가 가장 최근의 `commit`을 지정, conflict가 발생하는가를 확인

   - 따라서 `add`, `commit` 이후 `pull`을 하게 되면 `auto merge`가 수행됨

     **동일 파일을 작업하지 않았다는 전제 하에 가능한 작업!**

   - `auto merge` 이후에는 추가 작업 없이 `pull` 바로 수행

   - `git log`를 확인해 보면, `commit`의 시간 순서로 기록

     (먼저, 이후, merge 순서로 log 기록)

     **동일 파일을 작업하지 않으며, 작업자 간 소통을 많이 하는 것이 중요!!**

3. `Conflict` 발생 시

   - Github이 알아서 `merge` 해줄 수 있지만, 불가능한 상황에서는 직접 `merge` 작업 수행

     `add`, `commit` 후 `pull`을 당겨오면 불가능한 상황에 대해서는 `conflict`를 표시

   - VS Code는 초록색은 충돌한 나의 코드블록, 아래의 파란색 코드블록은 다른 사람의 충돌 영역을 표시

     - `Accept Current Change`

       내 code만 유지

     - `Accept Incoming Change`

       상대 code만 유지

     - `Accept Both Changes`

       둘 다 유지

   - conflict 해결 후 다음과 같이 `commit message`를 작성하는 것이 관례

     **`$ git commit -m "merge conflict"`**

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
