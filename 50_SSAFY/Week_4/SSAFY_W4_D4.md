# SSAFY Week4 Day4

 **참고자료** : ./50_SSAFY/8ython/notes/08.OOP_advanced.ipynb

---

## Git Co-Working

- git info 열람

  - `ls .git`
  - `cat .git/config`

- co-working 방법 두 가지

  1. 시간을 분리
  2. 파일을 분리

- Team Leader는 작업을 분배

  ---

- github new `organization`

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

---

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
   
   ---

- 과정

  - Status

    어떤 file을 `add`하여 Staging Area에 update 할 것인지 결정 후 `add`

  - Staging Area

    `add` 된 file들을 remote repository에 update 할 것인지 결정 후 `commit`

    **`commit` 한 내용들은 `git log`에서 확인 가능**하며,

    언제든 `git checkout` 혹은 `git reset`을 이용하여 해당 시점의 local state로 변경 가능

    - `commit` 은 **logical한 개발 단위**이므로, 중간 중간에 `commit`을 진행(ex. 기능 단위)

  - Commit Log

    staging area에서 `commit`한 내용들을 누적하여 기록

    ---

- 리베이싱

  실수로 빠뜨린 어떤 내용을 같이 커밋하고 싶다 할 때 사용

  commit을 다시 하여 합침

---

## mater / branch

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

  

---

- `git reset --hard 'log의 commit hash code'`

  해당 시점으로 **완전 변경**하며, 해당 commit 시점 이후의 모든 기록을 삭제(**복원 불가**)

- `git checkout --hard 'log의 commit hash code'`

  해당 시점으로 local을 변경

---

- `.gitignore`는 자신의 directory를 기준으로, 모든 하위 directory까지 영향
  - `__pycache__` : 같은 이름의 file, directory 무시
  - `__pycache__/*` : 내용물을 ingnore
  - `.gitignore` 파일도 upload

---

- GIPHY API

  .gif 짤 검색 사이트

  - GIPHY API 사용하여 짤 불러오기