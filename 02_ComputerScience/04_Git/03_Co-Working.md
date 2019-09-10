# Co-Working

## Github 기타 기능

### Organization

github에서 제공하는 gitlab의 `group` 기능

- 만약 owner가 아닌 member로 설정되어 있을 경우,

  repository 별로 member의 권한을 설정해줘야함채

<br>

### Issues

- 게시판 느낌으로 `New issue` 후 댓글 달듯이 Write를 통해 `@member` 설정하여, issue를 꼭 봐야하는 사람을 태그
- issue 해결 후 `closed`를 통해 종료
- `reopen` 도 가능

<br>

### mini-Trello 기능

Repository > `Projects` 탭

Template도 존재(Kanban)

<br><br>

## Collaboration

1. Push & Pull

   동기적 처리를 해야하는 업무

   ex) 끝말잇기 처럼 한 업무가 끝나야만 다음 업무를 수행할 수 있는 경우

   - 동시적 업무가 불가

2. **Branching & Pull**

   **현실 협업 모델**

3. Fork & Pull Request

   Open Source 혹은 사내 Code Contribution에 사용

<br><br>

## Branching & Pull 실습과정

### Repository Collaborator 권한 설정

1. github repository > `settings` > `branch protection rule` > `require pull request reviews before merging` + `Require review from Code Owner`
2. `Branch name pattern`에  'master' 기입하여 master 브랜치에 적용

<br>

### Branch 만들어 작업 후 Merge

1. 각자 master로부터 branch를 만들기

2. 한명은 md 파일, 한명은 html 파일을 수정하여 branch를 push

3. master로부터 다른 파일을 수정한 각각의 branch가 있을 때,

   하나씩 `Compare & pull request` 진행

4. 위에서의 권한 설정으로 인해 Administrator, collaborator 모두 review와 승인 과정 존재

   - Administrator의 request 또한 review 과정이 존재하며, merge는 언제나 가능
   - Collaborator는 administrator의 Approve를 받아야 merge 가능

5. merge가 끝난 Branch는 대체로 삭제

<br>

### Remote Repository의 Branch 정보 확인하기

- Remote repository 의 branch 정보 확인

  ```shell
  $ git branch -r
  ```

- Prune

  remote repository의 branch정보와 local에 저장된 remote branch 정보를 일치시키기

  (`git pull`은 단순히 code만 가져오므로, 해당 정보까지 동기화되지 않음)

  ```shell
  $ git remote prune origin
  ```

