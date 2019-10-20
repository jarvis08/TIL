# Update & Delete

## DELETE

### 테이블 삭제

```shell
DROP TABLE [TableName]
```

<br>

### 행, 레코드 삭제

```
DELETE FROM [TableName] WHERE [Condition];
```

```shell
sqlite> SELECT * FROM classmates;
id          name        age         address
----------  ----------  ----------  ----------
1           김퓨전         30          서울
5           김퓨전         30          서울
6           조동빈         28          서울

# 특정 레코드 삭제
sqlite> DELETE FROM classmates WHERE rowid=5;
sqlite> SELECT * FROM classmates;
id          name        age         address
----------  ----------  ----------  ----------
1           김퓨전         30          서울
6           조동빈         28          서울
```

SQLite 는 기본적으로 일부 행을 삭제하고 새 행을 삽입하면, 삭제 된 행의 값을 재사용하려고 시도한다. 즉 위에서 삭제한 5번 값을 재사용하게 된다. 하지만, 재사용할 경우 문제가 발생할 수 있다.

- 재사용하지 않게 하는 방법

  스키마 정의할 때 `PRIMARY KEY` column에 `AUTOINCREMENT`를 부여

  ```shell
  sqlite> CREATE TABLE tests(
  		id INTEGER PRIMARY KEY AUTOINCREMENT
  		name TEXT NOT NULL
  		);
  
  # 데이터 생성
  sqlite> INSERT INTO tests (name) VALUES ('홍길동');
  sqlite> INSERT INTO tests (name) VALUES ('임꺽정');
  sqlite> SELECT * FROM tests;
  1|홍길동
  2|임꺽정
  
  # 2번 데이터를 삭제 후 데이터를 추가하여도, 2번 key를 더이상 사용하지 않음을 확인 가능
  sqlite> DELETE FROM tests WHERE id=2;
  sqlite> INSERT INTO tests (name) VALUES ('왕건');
  sqlite> SELECT * FROM tests;
  1|홍길동
  3|왕건
  
  ```

<br><br>

## Update

### 데이터 수정하기

```shell
UPDATE [TableName] SET [ColumnName]=[변경 후 값] WHERE [ColumnName]=[변경할 항목]
```

```shell
sqlite> SELECT * FROM tests;
id          name
----------  ----------
1           홍길동
3           왕건

sqlite> UPDATE tests SET name='조동빈' WHERE id=1;
sqlite> SELECT * FROM tests;
id          name
----------  ----------
1           조동빈
3           왕건
```

<br>

### 테이블명 변경하기

```
ALTER TABLE [변경 전 테이블명] RENAME TO [변경 후 테이블명];
```

```shell
ALTER TABLE users2 RENAME TO users;
```

<br>

### 열, 필드 추가하기

```
ALTER TABLE [TableName] ADD COLUMN [ColumnName] DATATYPE;
```

```shell
sqlite> ALTER TABLE users ADD COLUMN created_at DATETIME NOT NULL;
Error: Cannot add a NOT NULL column with default value NULL
```

기존 데이터에 `NOT NULL` 조건으로 인해 `NULL` 값으로 새로운 컬럼이 추가될 수 없으므로 아래와 같은 에러가 발생한다. 따라서 `NOT NULL` 조건을 없애거나 기본값(`DEFAULT`)을 지정해야 한다.

따라서,

1. `NOT NULL` 조건 제거 후 추가

   ```shell
   sqlite> ALTER TABLE users ADD COLUMN created_at DATETIME; 
   ```

2. `DEFAULT` 부여

   ```shell
   sqlite> ALTER TABLE users ADD COLUMN gender TEXT NOT NULL DEFAULT 'male';
   998,"시우","고",15,"제주특별자치도",016-3732-8726,270,male
   999,"성호","최",28,"충청북도",010-5772-9832,6700,male
   1000,"예은","장",31,"경기도",016-5653-5019,8400,male
   ```