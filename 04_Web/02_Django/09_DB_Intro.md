# Database Intro

## Database란

- Databse

  체계화된 데이터의 모임

> Wikipedia
>
> 여러 사람이 공유하고 사용할 목적으로 통합 관리되는 정보의 집합
>
> 논리적으로 연관된 하나 이상의 자료의 모음
>
> 내용을 고도로 구조화함으로써 검색과 갱신의 효율화를 야기
>
> 몇 개의 자료 파일을 조직적으로 통합하여 자료 항목의 중복을 제거
>
> 자료를 구조화하여 기억시켜 놓은 자료의 집합체

- DB의 구성 요소

  - 개체, Entity

  - 속성, Attribute

    개체가 가지는 속성

  - 관계, Relation

    개체 사이의 관계

- RDBMS, 관계형 데이터베이스 관리 시스템

  관계형 모델을 기반으로하는 데이터베이스 관리시스템

  Oracle이 시장 지배, AWS에서 잡기 위해 Redshift를 운영

  - 대표적인 프로그램
    - 오픈소스: MySQL, SQLite, PostgreSQL
    - 유료: ORACLE, MS SQL

  - 데이터를 속성-컬럼(Attribute)과 데이터 값-레코드(Attribute Value)로 구조화한 2차원 테이블형태
  - 속성(Attribute)과 데이터 값(Attribute Value) 사이의 관계(Relation)을 찾아내고 이를 테이블 모양의 구조로 도식화 한다는 것을 의미

- SQLite

  Android는 실제로  SQLite를 사용

  가벼우며, 파일 단위로 데이터를 저장

---

## Terminology

### Database의 개념

- 스키마(Scheme)

  DB에서 data의 구조, 표현 방법, 관계 등을 정의한 구조

  DB의 **구조**와 **제약 조건**(자료의 구조, 표현 방법, 관계)에 관련한 전반적인 **명세**를 기술

  DB에서의 Metadata

  | column | datatype |
  | ------ | -------- |
  | id     | INT      |
  | age    | INT      |
  | phone  | TEXT     |
  | email  | TEXT     |

- 테이블(Table)

  열(column/field)과 행(record/value)의 모델을 사용하여 조직된 **데이터 요소들의 집합**

  SQL DB에서는 테이블을 '관계'라고도 한다.

- 열(Column)

  각 열에는 고유한 데이터 형식(INTEGER, TEXT, NULL 등)이 지정됨

- 행(Row), Record, Tuple

  테이블의 데이터는 행에 저장

  다음은 2개의 데이터가 저장된 테이블

  | id   | name | age  |
  | ---- | ---- | ---- |
  | 1    | hong | 42   |
  | 2    | kim  | 16   |

- 기본키(Primary Key)

  각 행(레코드)의 고유값

  반드시 설정되어야 하며, DB 관리 및 관계 설정 시 주요하게 활용

### SQL, Structured Query Laguage

- Wikipedia

  - 관계형 데이터베이스 관리시스템(RDBMS)의 데이터를 관리하기 위해 설계된 특수 목적의 프로그래밍 언어
  - 관계형 데이터베이스 관리 시스템에서 자료의 검색과 관리 데이터베이스 스키마 생성과 수정, 데이터베이스 객체 접근 조정 관리를 위해 고안

- SQL을 이용하여 Data를 조작

  `Database <-SQL-> Code`