# Git Commands

## Making a directory a Git Repository

- `$ git init`

  : 기존 프로젝트를 git으로 관리하고 싶을 때, 프로젝트의 디렉토리로 이동해서 명령

  .git이라는 하위 디렉토리 생성

  저장소에 필요한 뼈대 파일(skeleton)이 생성됨

- `$ git add 파일/디렉토리명`

- `$ git commit -m 'Commit Comments'`

  

---

## Cloning a Git Repository

- `$ git clone git://github.com/schacon/grit.git`

  grit이라는 디렉토리를 생성하며 repository 내용을 clone

- `$ git clone git://github.com/schacon/grit.git mygrit`

  mygrit이라는 디렉토리를 생성하여 repository 내용을 clone

- Git은 다양한 프로토콜을 지원한다. 이제까지는 `git://` 프로토콜을 사용했지만 `http(s)://`를 사용할 수도 있고 `user@server:/path.git`처럼 SSH 프로토콜을 사용할 수도 있다

---

