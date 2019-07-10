# Python

---

## Turing Completeness 

- 저장 :: 어디에 무엇을 어떻게 넣는가?
  1. 숫자
  2. 글자
  3. 참/거짓
- 조건
- 반복

------

## Python은 어떻게 저장하는가?

- 변수 (variable) 

  :: 박스 1개

- 리스트 (list)

  :: 박스 여러개

  ​	list = ["강남구", ...]

- 딕셔너리 (dictionary)

  :: 견출지 붙인 박스들의 묶음

  ​	dic = {"강남구" : 50, ...}

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

## requests, bs4

`requests.get( 'url' )` : http status code

`requests.get( 'url' ).text` : url로부터 document를 text 형태로 받음

`requests.get( 'url' ).json()` : url로부터 document를 json 형태로 받음

`bs4.BeautifulSoup(response,'xml')` : get()의 내용을 저장한 response 변수의 xml 타입을 파이썬이 보기 좋은 형태로 변환

`document = bs4.BeautifulSoup(response, 'html.parser')` : response를 html parser를 사용하여 변형

`document.select('.ah_k', limit=10)` : document의 ah_k class 중 10개 고르기

`document.select_one(selector).text` : css selector 중 하나를 text화

---

## webbroser, flask

```python
import webbrowser

url = "https://search.daum.net/search?q="
keywords = ["수지", "한지민"]
for keyword in keywords:
    webbrowser.open(url + keyword)
```

`flask run` : flask 서버 구동

http://naver.com:80/***상세주문***

- 주문서

  1. 어떻게 제공

     `"/"` : CS에서 / 의미는 root

     `@app.route("/")` : 가장 상위를 의미

  2. 무엇을 제공

```python
from flask import Flask
app = Flask(__name__)
# 플라스크를 코드 안에 불러옴


### 1. 주문 받는 방식(어떻게) - @app.route("/")
"""
CS에서 /는 root를 의미
따라서 가장 상위를 의미
"""
### 2. 무엇을 제공할지 - def hello():
@app.route("/")
def hello():
    return "Hello World!"

@app.route("/hi")
def hi():
    return "hi"
# 주문 방식과 제공함수명은 일치시키는게 좋음

# 1. /name
# 2. 응답 : 영문 이름
@app.route("/name")
def name():
    return "Dongbin Cho"

# 이름을 붙여서 인사해주기
@app.route("/hello/dongbin")
def hello_1(name):
    return "Hello, {}".format(name)

# 사람에 따라 이름을 붙여서 인사해주기
# person과 같은 명칭은 내가 정의
@app.route("/hello/<person>")
def hello_person(person):
    return "Hello, {}!".format(person)
    # return f"Hello, {person}!"
    # return "Hello, } + person
```

- 포트

  : 접속 경로(문)

  http - 80, https - 443 포트를 주로 정문으로 사용(외부 사용자용)

  자기 자신으로의 접속은 22

- Routing

  : 주문이 왔을 때, 답변을 하는 경로

- Variable Routing

  : 사용자의 요청이 왔을 때, 변수에 따라 가변적으로 응답

  ```python
  @app.route("/hello/<person>")
  def hello_person(person):
      return "Hello, {}!".format(person)
      # return f"Hello, {person}!"
      # return "Hello, } + person
  ```

- Flask에서 template 사용하기

  `return render_template('파일명.html', html변수명=py변수명)`

  ```python
  @app.route("/hello/<name>")
  def hello(name):
      return render_template('hello.html', tem_name=name)
      # tem_name :: template 안에서 사용할 변수명
      # template에서는 <!-- --> 주석 사용 불가
      # html이 아닌 template의 문법으로 처리
  ```

- 서버 재시동 하지 않아도 되게 하기

  ```python
  if __name__ == "__main__":
      app.run(debug=True)
  # 이후 터미널에서 'python 파일명.py' 명령으로 실행
  ```

- Flask 구조

  dynamic directory(app.py) > templates directory (htmls)

  app.py는 templates directory 안의 html template을 이용하여 요청에 응답