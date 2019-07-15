# Python

---

## String

```python
print('touch example_{}.txt'.format(i))
print(f'touch example_{i}.txt')
print('touch example'+ str(i) + '.txt')
```

`string.replace('변경 전', '변경 후')` : string의 일부를 원하는 글자로 수정

---

## List

`list.reverse()` : list 원본의 순서를 역으로 변환

---

## Dictionary

`list(dic.keys())` : dictionary의 key 값들을 list로 변환

`dic.get(key).get(key)` : 해당 key의 value를 반환, 없을 시 error 대신 None 반환

---

## Set

: set은 중복 요소가 없으며, 오름차순

`set( list )` : list를 집합으로 전환

```python
count = len(set(winner) & set(ur_lotto))
# winner list와 ur_lotto list를 비교할 때
# for문을 이용하는 것 보다 빠른 속도로 같은 요소의 개수를 구함
```

---

## Sort

`sorted([ ])` : list를 오름차순으로 sort

---

## File r / w / a

`with open('파일명', '파일 조작 유형', encoding='utf-8') as f:`

`f.readlines()` : 모든 문장 읽기

`f.readline()` : 한 줄 읽기

`f.write()` : 한 번 쓰기

`f.writelines()` : 모두 쓰기

- 파일 조작 3가지

  `'r'`: read

  `'w'`: write

  `'a'`: append

---

## os

`os.listdir()`: 현재 디렉토리 내부의 모든 파일, 디렉토리를 리스트에 저장

`os.rename(현재 파일명, 바꿀 파일명)`: 파일명 변경

`os.system()`: Terminal에서 사용하는 명령어 사용

```shell
os.system('touch example.txt')
# example.txt 파일 생성
os.system('rm example.txt')
# example.txt 파일 제거
```

`os.chdir()`: 작업 폴더를 현 위치에서 해당 위치로 옮김

---

## random

`random.sample( [], int )` : [ ] 중 int 개 만큼 비복원 추출

`random.choice( [ ] )` : [ ] 중 1개를 임의 복원 추출

---

## decouple

:: key 암호화시키기

- directory 안에 `.env` 파일 생성(linux에서는 .으로 시작하면 숨김파일)

- 모두 대문자로 작성

```python
# .env 파일 내부에 아래와 같이 작성하며, .env파일은 공유되지 않아야한다.
TELEGRAM_TOKEN = "토큰 정보 기입"
# 이를 위해 .gitignore 파일을 생성하고, 무시하고자 하는 파일명을 기입
.env
```

```python
# token url 사용할 때
from ducouple import config
# token_url을 원래 토큰 값 대신 config('.env에 작성한 token을 넣은 변수명')
token_url = config('TELEGRAM_TOKEN')
```
