# SQLite Setup

다운로드 페이지: [SQLite Download Page](https://www.sqlite.org/download.html)

1. 다운로드 페이지에서 OS에 맞는 버전의 dll 파일과 tools 파일 다운로드

2. 설치 위치 지정

   ```shell
   $ cd ~/
   $ mkdir splite
   # sqlite에 위에서 다운받은 dll 압축 파일의 내용과 tools 압축 파일의 내용을 모두 옮겨옴
   ```

3. 실행

   - 방법 1

     옮겨온 상태에서 exe 파일 실행

   - 방법 2

     Path에 위에서 설정한 `sqlite`를 추가

     git bash는 윈도우에서 시그윈(Cygwin)과 winpty를 사용하는데, 윈피티를 사용하여 sqlite3를 실행해야 한다.

   ```shell
   $ winpty sqlite3
   ```

4. DB 실행

   ```shell
   $ cd 파일루트
   $ winpty sqlite3 db.sqlite3
   SQLite version 3.29.0 2019-07-10 17:32:03
   Enter ".help" for usage hints.
   sqlite>
   ```

5. DB 작업

   모든 명령어는 `.`로 시작해야 하며, 쿼리문 뒤에는 세미콜론(`;`) 작성

6. 작업 종료

   종료 명령어: `.exit`

   ```shell
   sqlite> .exit
   ```

- `alias`에  sqlite 명령어 추가하기

  ```shell
  $ cd ~/.bashrc
  alias='winpty sqlite3'
  ```

<br>

### SQLite 명령어

- csv 다루기

  `.mode csv`

  ```shell
  sqlite> .mode csv
  # hellodb.csv 파일을 examples라고 지정하는 테이블에 담아서 불러옴
  sqlite> .import hellodb.csv examples
  ```

- 조회 할 때 column 형태로 표시하기

  `.mode column`

  ```shell
  sqlite> SELECT * FROM tests;
  1|홍길동
  3|왕건
  
  sqlite> .mode column
  sqlite> SELECT * FROM tests;
  1           홍길동
  3           왕건
  
  ```

- column 이름 표기하기

  `.headers on`

  ```shell
  sqlite> .headers on
  sqlite> SELECT * FROM tests;
  id          name
  ----------  ----------
  1           홍길동
  3           왕건
  
  ```

<br>