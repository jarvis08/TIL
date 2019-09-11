# Webhook

- `GET` : 일반적인 내용을 요청
- `POST` : 암호화된 보안 중요사항을 요청

<br>

<br>

## Web Hook(Reverse API)

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

## webbroser

```python
import webbrowser

url = "https://search.daum.net/search?q="
keywords = ["수지", "한지민"]
for keyword in keywords:
    webbrowser.open(url + keyword)
```
