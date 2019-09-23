# Terms

## Basic Terms

| 용어                    | 설명                                                         |
| ----------------------- | ------------------------------------------------------------ |
| 스키마, Schema          | 데이터베이스의 구조와 제약 조건(자료의 구조, 표현 방법, 관계)에 관련한 전반적인 명세를 기술한 것 |
| 테이블, Table           | 열(컬럼/필드)과 행(레코드/값)의 모델을 사용해 조직된 데이터 요소들의 집합.<br/>SQL 데이터베이스에서는 테이블을 관계 라고도 한다. |
| 열(Column), 필드(Field) | 각 열에는 고유한 데이터 형식이 지정된다.                     |
| 행(Row), 레코드(Record) | 테이블의 데이터는 행에 저장된다.<br>즉, user 테이블에 4명의 고객정보가 저장되어 있으며, 행은 4개가 존재한다. |
| 기본키, Primary Key     | 각 행(레코드)의 고유값으로 Primary Key로 불린다.<br/>반드시 설정하여야하며, 데이터베이스 관리 및 관계 설정시 주요하게 활용된다. |

<br>

### SQL, Structured Query Language

위키피디아

> 관계형 데이터베이스 관리시스템(RDBMS)의 데이터를 관리하기 위해 설계된 특수 목적의 프로그래밍 언어이다.  관계형 데이터베이스 관리 시스템에서 자료의 검색과 관리, 데이터베이스 스키마 생성과 수정, 데이터베이스 객체 접근 조정 관리를 위해 고안되었다. 

SQL 문법은 다음과 같이 세가지 종류로 구분될 수 있다.

| 분류                                                     | 개념                                                         | 예시                                        |
| -------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------- |
| DDL - 데이터 정의 언어<br/>(Data Definition Language)    | 데이터를 정의하기 위한 언어이다.  <br/>관계형 데이터베이스 구조(테이블, 스키마)를  <br/>정의하기 위한 명령어이다. | CREATE <br/>DROP <br/>ALTER                 |
| DML - 데이터 조작 언어 <br/>(Data Manipulation Language) | 데이터를 저장, 수정, 삭제, 조회 등을 하기 위한 언어이다.     | INSERT <br/>UPDATE <br/>DELETE <br/>SELECT  |
| DCL - 데이터 제어 언어 <br/>(Data Control Language)      | 데이터베이스 사용자의 권한 제어를 위해 사용되는 언어이다.    | GRANT <br/>REVOKE <br/>COMMIT <br/>ROLLBACK |

<br>

### Database 자료형

SQLite은 동적 데이터 타입으로, 기본적으로 유연하게 데이터가 들어간다. 
BOOLEAN은 정수 0, 1 으로 저장된다.

명령어는 소문자로 사용할 수 있지만, 대문자로 사용하는 것이 Convention이다.

`INTEGER`는 `INT`로도 사용할 수 있으며, 둘 다 별명으로써 같은 '정수'를 할당하는 기능에 묶여있다.

(하지만, **Primary Key에 설정하려면 반드시 `INTEGER`를 사용**해야한다!)

| Affinity                   |                                                              |
| -------------------------- | ------------------------------------------------------------ |
| `INTEGER`<br>혹은<br>`INT` | TINYINT(1byte), SMALLINT(2bytes), MEDIUMINT(3bytes), INT(4bytes), BIGINT(8bytes), UNSIGNED BIG INT |
| `TEXT`                     | CHARACTER(20), VARCHAR(255), TEXT                            |
| `REAL`                     | REAL, DOUBLE, FLOAT                                          |
| `NUMERIC`                  | NUMERIC, DECIMAL, BOOLEAN, DATE, DATETIME                    |
| `BLOG`                     | no datatype specified                                        |