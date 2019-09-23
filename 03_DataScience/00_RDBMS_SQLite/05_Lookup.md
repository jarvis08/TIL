# Lookup

기본적인 조회 방법

```
SELECT [ColumnName] FROM [TableName]
```

- 나이가 30이상이며 성이 김씨인 사람들의 성과 이름을 출력하기

  ```shell
   sqlite> SELECT last_name, age FROM users WHERE age >= 30 AND last_name='김';
  ```

<br>

### 출력 개수 제한

개수 제한하기

```
SELECT * FROM [TableName] LIMIT [개수];
```

**오프셋(Offset)**을 통해 출력 시작 시점 조작하기

- 사용사례) 게시판 페이징하여 끊어서 출력하기

  `LIMIT 50 OFFSET 50`일 경우 51번 부터 50개

```
SELECT * FROM [TableName] LIMIT [개수] OFFSET [개수];
```

예시

```shell
sqlite> SELECT * FROM classmates LIMIT 2;
id          name        age         address
----------  ----------  ----------  ----------
1           김퓨전         30          서울
5           김퓨전         30          서울

# 
sqlite> SELECT * FROM classmates LIMIT 2 OFFSET 1;
id          name        age         address
----------  ----------  ----------  ----------
5           김퓨전         30          서울
6           조동빈         28          서울

# 세 번째 값만 콕찝어 보기
sqlite> SELECT * FROM classmates LIMIT 1 OFFSET 2;
id          name        age         address
----------  ----------  ----------  ----------
6           조동빈         28          서울
```

<br>

<br>

## 데이터 조회

### 특정 값을 보유한 Column 조회

`WHERE` 사용

```
SELECT * FROM [TableName] WHERE [ColumnName]=[Value];
```

```shell
sqlite> SELECT * FROM classmates WHERE id=5;
id          name        age         address
----------  ----------  ----------  ----------
5           김퓨전         30          서울
```

<br>

### 중복 없이 가져오기

```
SELECT DISTINCT [ColumnName] FROM [TableName];
```

<br>

### Count, 숫자 세기

```shell
SELECT COUNT([ColumnName]) FROM [TableName];
```

<br>

### 평균/총합/최대/최소

```
SELECT AVG([ColumnName]) FROM [TableName];
```

```
SELECT SUM([ColumnName]) FROM [TableName];
```

```
SELECT MAX([ColumnName]) FROM [TableName];
```

```
SELECT MIN([ColumnName]) FROM [TableName];
```

```shell
sqlite> SELECT first_name, MAX(balance) FROM users;
first_name,MAX(balance)
"선영",990000
```

<br>

### 패턴 적용

`LIKE` 정확한 값에 대한 비교가 아닌, 패턴을 확인하여 해당하는 값을 반환

- `_`: 반드시 이 자리에 한 개의 문자가 존재해야 한다.
- `%`: 이 자리에 문자열이 있을 수도, 없을 수도 있다.

| 표현 | 예시           | 설명                                            |
| ---- | -------------- | ----------------------------------------------- |
| `%`  | `2%`           | 2로 시작하는 값                                 |
|      | `%2`           | 2로 끝나는 값                                   |
|      | `%2%`          | 2가 들어가는 값                                 |
| `_`  | `_2%`          | 아무 값이나 들어가며, 두 번째가 2로 시작하는 값 |
|      | `1___`         | 1로 시작하며, 네 자리인 값                      |
|      | `2_%_% / 2__%` | 2로 시작하며, 적어도 세 자리인 값               |

<br>

### 정렬

- ASC: 오름차순
- DESC: 내림차순

```
SELECT [ColumnName] FROM [TableName] ORDER BY [ColumnName] [정렬방법];
```

```shell
sqlite> SELECT * FROM users ORDER BY age ASC LIMIT 10;
id,first_name,last_name,age,country,phone,balance
11,"서영","김",15,"제주특별자치도",016-3046-9822,640000
59,"지후","엄",15,"경상북도",02-6714-5416,16000
61,"우진","고",15,"경상북도",011-3124-1126,300
125,"우진","한",15,"강원도",011-8068-4814,3300
144,"은영","이",15,"전라남도",010-5284-4904,78000
196,"지훈","김",15,"전라북도",02-9385-7954,760
223,"승현","장",15,"충청북도",016-5731-8009,450
260,"주원","김",15,"전라남도",02-4240-8648,6300
294,"은정","이",15,"경상북도",010-6099-6176,5900
295,"정수","강",15,"충청북도",02-7245-5623,500
```

