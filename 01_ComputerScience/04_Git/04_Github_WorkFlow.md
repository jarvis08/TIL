# Github Work Flow

## 기본 작업

### 1. 내 Remote/Local Repository로 가져와서 작업

1. 개발자는 작업하고자 하는 관리자의 remote repository를 자신의 remote repository로 `Fork`
2. `Fork` 해온 자신의 remote repository를 `clone`하고, local repository로 가져와서 작업
3. local 작업 후 `Fork` 해왔던 자신의 remote repository에 `Push`

<br>

### 2. 관리자의 Remote Repository에 Merge

1. 개발자의 remote repository에서 `This branch is 1 commit ahead of Hoony104:master.`라는 메세지를 확인 가능
2. 내 remote repository의 수정된 코드를 `Fork`했던 관리자의 원본 remote repository에 적용시키기 위해 `New pull request` > `Create pull request`를 요청
3. 관리자는 github에서 `Merge pull request` 기능을 사용하여 `merge`

<br>

<br>

## 갱신하기

관리자의 remote repository가 변경되었다 해도, `Fork` 해온 나의 remote repository는 갱신되지 않으며, 내 remote repository에 갱신 작업을 해 주어야 한다.

갱신 방법에는 두가지가 있다.

<br>

### 1. 다시 Fork

가장 쉬운 방법이지만, 사용하고 싶지 않다.

<br>

### 2. Fetch 사용하기

`Pull = Fetch + Merge`이며, 생략되어 사용되어 온 `fetch`를 사용 할 때이다. 전체 과정은 다음과 같다.

1. Fork한 나의 remote repository 보다 나의 **local repository를 먼저 동기화**

   ```shell
   $ git remote add upstream 관리자repo주소
   $ git fetch upstream
   ```

   ```shell
   # 결과
   remote: Enumerating objects: 9, done.
   remote: Counting objects: 100% (9/9), done.
   remote: Compressing objects: 100% (4/4), done.
   remote: Total 7 (delta 1), reused 6 (delta 1), pack-reused 0
   Unpacking objects: 100% (7/7), done.
   From https://github.com/sspy21/nhaeng
    * [new branch]      master     -> upstream/master
   ```

   이후 merge 시도

   ```shell
   $ git merge upstream/master
   Auto-merging README.md
   CONFLICT (content): Merge conflict in README.md
   Automatic merge failed; fix conflicts and then commit the result.
   ```
   이후 Incoming chance를 accept하여 변경된 내용으로 덮어쓴다.
   
   conflict 해결 후 `add`, `commit`
   
   ```shell
   $ git add .
   $ git commit -m "fix merge conflict"
   ```

2. 나의 remote repository로 push한 후 pull request

   



