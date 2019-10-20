# Schema

### 타이트하게 스키마 제작하기

1. Null 값을 허용하면 안되며

2. Primary Key를 지정해야 한다.

   Primary Key는 INTEGER column에만 가능하다.

```shell
sqlite> CREATE TABLE classmates(
   ...> id INTEGER PRIMARY KEY,
   ...> name TEXT NOT NULL,
   ...> age INTEGER NOT NULL,
   ...> address TEXT NOT NULL
   ...> );

# NOT NULL을 선언하였으며, 기본키를 지정하도록 했으므로 값의 개수가 부족하면 저장할 수 없다.
sqlite> INSERT INTO classmates VALUES('김퓨전', 30, '서울');
Error: table classmates has 4 columns but 3 values were supplied

# NOT NULL 설정 시 빈 값으로 넣을 수 없다.
sqlite> INSERT INTO classmates (age, address) VALUES (30, '부산');
Error: NOT NULL constraint failed: classmates.name

sqlite> INSERT INTO classmates VALUES(1, '김퓨전', 30, '서울');
sqlite> SELECT * FROM classmates;
id          name        age         address
----------  ----------  ----------  ----------
1           김퓨전         30          서울
```

마음대로 Primary Key 지정해보기

```shell
sqlite> INSERT INTO classmates VALUES(5, '김퓨전', 30, '서울');
sqlite> SELECT * FROM classmates;
id          name        age         address
----------  ----------  ----------  ----------
1           김퓨전         30          서울
5           김퓨전         30          서울
```

자동으로 Primary Key 할당하기

```shell
sqlite> INSERT INTO classmates (name, age, address) VALUES ('조동빈', 28, '서울');
sqlite> SELECT * FROM classmates;
id          name        age         address
----------  ----------  ----------  ----------
1           김퓨전         30          서울
5           김퓨전         30          서울
6           조동빈         28          서울

# 자동으로 생성되는 row id 또한 지정한 primary key값을 부여받는다.
sqlite> SELECT rowid, * FROM classmates;
id          id          name        age         address
----------  ----------  ----------  ----------  ----------
1           1           김퓨전         30          서울
5           5           김퓨전         30          서울
6           6           조동빈         28          서울

```

<br><br>

## django Model을 통해 스키마 확인

```shell
$ cd ~/00_jarvis08/02_Django/BOARD/
$ sqlite3 db.sqlite3
SQLite version 3.29.0 2019-07-10 17:32:03
Enter ".help" for usage hints.

# DB의 테이블 목록 확인
sqlite> .tables
auth_group                  django_admin_log
auth_group_permissions      django_content_type
auth_permission             django_migrations
auth_user                   django_session
auth_user_groups            posts_comment
auth_user_user_permissions  posts_post

# Post 모델 확인
sqlite> .schema posts_post
CREATE TABLE IF NOT EXISTS "posts_post" ("id" integer NOT NULL PRIMARY KEY AUTOI
NCREMENT, "title" varchar(100) NOT NULL, "content" text NOT NULL, "image" varcha
r(100) NOT NULL, "created_at" datetime NOT NULL, "updated_at" datetime NOT NULL)
;
```

- `CREATE TABLE IF NOT EXIST`: 존재하지 않을 경우에만 객체를 새로 생성
- `varchar(100)`: models.py에서 `max_length`를 지정했으므로, 변경되는 사항을 적용하는 `VARCHAR` 사용
- models.py에서는 `image` column의 경우 `None` 값을 허용하도록 되어있지만, **DB의 경우 `NULL` 값은 굉장히 피곤한 존재이므로 `NOT NULL`을 적용 후 따로 관리**