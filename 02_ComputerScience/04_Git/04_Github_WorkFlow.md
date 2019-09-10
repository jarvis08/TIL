# Github Work Flow

Project Repository의 `Push` 권한이 없을 때

<br>

### 1. 내 Remote/Local Repository로 가져와서 작업

1. 개발자는 작업하고자 하는 관리자의 remote repository를 자신의 remote repository로 `Fork`
2. `Fork` 해온 자신의 remote repository를 `clone`하고, local repository로 가져와서 작업
3. local 작업 후 `Fork` 해왔던 자신의 remote repository에 `Push`

<br>

### 2. 관리자의 Remote Repository에 Merge

1. 개발자의 remote repository에서 `This branch is 1 commit ahead of Hoony104:master.`라는 메세지를 확인 가능
2. 내 remote repository의 수정된 코드를 `Fork`했던 관리자의 원본 remote repository에 적용시키기 위해 `New pull request` > `Create pull request`를 요청
3. 관리자는 github에서 `Merge pull request` 기능을 사용하여 `merge`

