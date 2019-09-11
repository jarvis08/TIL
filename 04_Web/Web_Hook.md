# Web

## Webhook

- `GET` : 일반적인 내용을 요청

- `POST` : 암호화된 보안 중요사항을 요청

- Web Hook(Reverse API)

  상태변화의 발생을 캐치하여 반응할 수 있도록 함

  Telegram도 web hook 기능 제공

  ```python
  ### webhook setup
  ## telegram이 메세지왔다고 우리한테 알림 해주는게 목적
  # 브라우저 주소창 :: telegram주소/bot+token/setWebhook?url=내주소/token
  'https://api.telegram.org/'+'bot'+token+'/'+'setWebhook?'+'url='+'https://7e5f168e.ngrok.io/'+ token
  # 내주소 뒤의 token은 .env에 기록한 내용과 완전히 동일해야함
  # telegram의 bot을 언급할 때에는 무조건 bot + token이 되어야 한다
  
  # webhook 해제 :: telegram주소/bot+token/deletewebhook
  'https://api.telegram.org/'+'bot'+token+'/'+'deletewebhook'
  ```

- Port : 접속 경로(문)

  (local - 22, http - 80, https - 443) 주로 사용

<br><br>

## ngrok

cmd >> ngork.exe 위치 >>`ngrok http 5000 + Enter`

:: 5000포트를 이용하여 외부 접속이 가능하도록 설정

:: 아래 코드에서는 # 처리해둔(내가 임의로 보기 쉬우라고 붙임), https://7e5f168e.ngrok.io 통해 접속 가능

```python
ngrok by @inconshreveable                                                                               (Ctrl+C to quit)

Session Status                online
Session Expires               7 hours, 59 minutes
Version                       2.3.30
Region                        United States (us)
Web Interface                 http://127.0.0.1:4040
Forwarding                    http://7e5f168e.ngrok.io -> http://localhost:5000
# Forwarding                    https://7e5f168e.ngrok.io -> http://localhost:5000

Connections                   ttl     opn     rt1     rt5     p50     p90
                              0       0       0.00    0.00    0.00    0.00
```

<br><br>

## requests, bs4

`requests.get( 'url' )` : http status code

`requests.get( 'url' ).text` : url로부터 document를 text 형태로 받음

`requests.get( 'url' ).json()` : url로부터 document를 json 형태로 받음

`bs4.BeautifulSoup(response,'xml')` : get()의 내용을 저장한 response 변수의 xml 타입을 파이썬이 보기 좋은 형태로 변환

`document = bs4.BeautifulSoup(response, 'html.parser')` : response를 html parser를 사용하여 변형

`document.select('.ah_k', limit=10)` : document의 ah_k class 중 10개 고르기

`document.select_one(selector).text` : css selector 중 하나를 text화

<br><br>

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

- 맨 아래 기입 시, 서버 켜둔 채 유지

  ```python
  if __name__ == "__main__":
      app.run(debug=True)
  # 이후 터미널에서 'python 파일명.py' 명령으로 실행
  ```

- Flask 구조

  dynamic directory(app.py) > templates directory (htmls)

  app.py는 templates directory 안의 html template을 이용하여 요청에 응답
  
- `get_json()`

  parameter 중 `silent` 파라미터는 JSON parsing fail 에 대해서 None 처리 여부를 설정 가능

  기본값을 False인데, 명시적으로 True 로 주면 호출시 에러가 나지 않고 None을 리턴

  ```python
  print request.get_json(silent=True) 
  ```
