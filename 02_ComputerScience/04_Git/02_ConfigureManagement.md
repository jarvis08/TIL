# 형상 관리, Configure Management

[git-visualizer](https://git-school.github.io/visualizing-git/)를 통해 git의 branch 생성 방식을 시각적으로 확인 가능

많은 기업에서 `master` branch는 service 영역으로 사용하며, 개발자들은 `develop` branch에서 개발한다. 개발 도중에는 `develop` branch에서 `feature login`, `feature createPost` 등과 같이 기능 별로 branch를 새로 구성하여 개발 후 `develop` branch에 `merge`한다. 기능 별 개발 완료 및 branch `merge` 이후에는 `master`와 `develop` branch를 제외하고는 삭제해준다. 기타 Branch들은 유지용이 아닌, 사용하고 버리는 용도로 사용된다.





## Master & Branch

- master에서 branch를 생성

  master와 branch가 서로 영향을 주지 않기 때문에 병행 개발이 가능

- 현업 예시

  - master = 제품
  - branch = 신기능



### Branch

Branch 정보를 확인, branch 생성/삭제 명령어

```shell
# branch 확인
$ git branch

# branch 생성
$ git branch [branchName]

# branch 삭제
$ git branch -d [branchName]
```



### Checkout

Head를 옮기는 명령어로, `checkout`을 이용하여 **과거/현재의 Commit** 혹은 **다른 branch**로 이동 가능

```shell
# branch로 checkout
$ git checkout [branchName]
# branch 생성 및 checkout
$ git checkout -b [branchName]
```



### Switch

branch로 이동 전용 명령어이며, 2.23 version 부터 사용 가능

```shell
# switch는 현재 시점으로 이동할 때 사용
$ git switch [branchName]
# Brach 새로 만들면서 이동
$ git switch -c [branchName]
```



### Merge

```shell
# 현재 branch에서 특정 브랜치인 [branchName]을 병합(흡수)
$ git merge [branchName]
```





## Merge Scenario

### Fast-Forward Merge, 빨리 감기 병합

- Fast-Forward Merge 예시

  혼자 공부 할 때, `master`의 내용을 유지한 채 branch를 이용하여 실험해본 후 `master`에서 `merge`

`WorkingBrach`가 `master` branch 보다 앞선 상태일 때, `master`에서 `merge [branchName]`를 진행

- 다음과 같은 과정을 Fast-Forward Merge라고 칭함

  1. `master`에서 branch 생성

  2. `master`에서는 이후 **아무런 `commit`이 없는 상태**

     - `master`에서 다른 가지(branch)가 생성되지 않은 상태

     - 1번에서 생성했던 branch 또한 `가지`가 아닌 `줄기`를 따라 전진하는 형태

  3. `branch`에서 작업 후 `commit`

  4. `master`에서 `git merge branch`



### Auto Merge

branch 간 다른 파일을 수정했을 경우

ex. `master` branch에서는  html 파일을 생성하고, `develop` branch에서는 md file을 수정

- merge 시도 시,
  1. 자동으로 vim을 이용하여 `merge branch` 문구가 생성하고 보여줌
  2. `:wq` 입력하면 자동으로 auto merge 진행하며 commit



### Merge with Conflict

ex) 동일 line에 branch 간 다른 내용을 기입한 후 merge 시도

1. Merge 해결하기

    - VS Code에서 merge하기

      3 가지 방법 중 선택

      1. Current Change

         병합의 주체(merge를 시도한 branch)

      2. Incoming Change

         병합시키는 대상

      3. Accept Both

    - 해당 merge 기능을 지원하지 Text Editor에서 해결하기

      아래 세 가지 문구를 삭제, 원하는 문구만 남긴 후 저장

      ```
      <<<<<<< HEAD
      =======
      >>>>>>> develop
      ```

2. commit 하기

   ```shell
   $ git commit -m "resolve merge conflict"
   ```





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

