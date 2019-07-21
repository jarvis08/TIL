# Python External Libraries

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

---

## CSV

- CSV = Comma Seperated Values
  - `csv.writer(파일명)` : '파일명'을 조작하는 writer를 원하는 이름으로 생성

    `writer.writerow(iterable_item)` : iterable_item을 전에 생성한 writer를 사용하여 한줄씩 끊어서 작성(writerow)

  - `csv.DictWriter(파일명, fieldnames=필드명_list)`

    이 함수를 선언시, `writer.writerow(iterable)`을 통해 dictionary를 그대로 넣어도 csv 형태 작성

    `writer.writeheader()` : 해당 코드를 넣어두면 fieldnames까지 첫 줄에 작성

  ```python
  # csv 파일 만들기
  lunch = {
      '진가와' : '01011112222',
      '대우식당' : '01054813518',
      '바스버거' : '01088465846'
  }
  # 1. lunch.csv 데이터 저장
  with open('lunch.csv', 'w', encoding='UTF-8') as f:
      for k, v in lunch.items():
          f.write(f'{k},{v}\n')
  
  # 2. ',' join을 사용하여 string 만들기
  with open('lunch.csv', 'w', encoding='UTF-8') as f:
      for item in lunch.items():
          f.write(','.join(item))
  
  # 3. csv 라이브러리 사용
  # writer와 reader가 따로 존재
  import csv
  with open('lunch.csv', 'w', encoding='utf-8', newline='') as f:
      csv_writer = csv.writer(f)
      for item in lunch.items():
          csv_writer.writerow(item)
          
  # 4. csv.DictWriter()
  # field name을 설정 가능
  with open('student.csv', 'w', encoding='utf-8', newline='') as f:
      fieldnames = ['name', 'major']
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerow({'name':'john', 'major':'cs'})
      writer.writerow({'name':'dongbin', 'major':'ie'})
  ```

