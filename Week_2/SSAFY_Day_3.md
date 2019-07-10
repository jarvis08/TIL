# SSAFY_Day_3

---

- MVP (Minimum Viable Product)

  : 최소 기능 제품

  가장 필요한 기능을 빠르게 구현하여 출시한 후, 계속해서 사용자의 니드를 보완

  반대 개념 - 완결성 있는 제품을 장기간에 걸쳐 개발하여 출시

---

## Web Server

`pip install flask`

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

  

```python
@app.route("/hello/<name>")
def hello(name):
    return render_template('hello.html', tem_name=name)
    # tem_name :: template 안에서 사용할 변수명
    # template에서는 <!-- --> 주석 사용 불가
    # html이 아닌 template의 문법으로 처리
```

- 수정 시 서버 재시동 하지 않아도 되게 하기!

  ```python
  # 방법 1
  if __name__ == "__main__":
      app.run(debug=True)
  ```

  

