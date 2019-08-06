# 형상 관리, Configure Management

---

## Mater & Branch

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

   - `Conflict` 발생 시

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

       **$ git commit -m "merge conflict"**