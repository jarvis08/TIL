# Webhook

- `GET` : 일반적인 내용을 요청

- `POST` : 암호화된 보안 중요사항을 요청

- Port : 접속 경로(문)

  (local - 22, http - 80, https - 443) 주로 사용

<br>

<br>

## Webhook(Reverse API)

Telegram의 `getUpdates` 메서드의 경우, 새 메세지가 올 때마다 `새로고침`(최신화) 해야만 그 메시지를 사용할 수 있다. 하지만 webhook의 경우 새 메세지가 올 때마다 지정한 저장소로 새 메세지에 대한 신호와 정보를 전달한다. 따라서 챗봇과 같은 자동화된 서비스에는 webhook 혹은 유사한 기능을 하는 요소가 필요하다.

간단히 정의하자면, webhook은 상태변화의 발생을 캐치하여 반응한다.

- Telegram도 webhook 기능 제공

  목적 : Telegram이 메세지왔다고 사용자에게 알림

  - Webhook 켜기

    `{Telegram주소}/bot{token}/setWebhook?url={내주소}/{token}`

    ex) https://api.telegram.org/bot123123/setWebhook?url=https://abcdefg/123123

    - Telegram의 bot을 언급할 때에는 `bot` + `token` 형태로 사용

    - 맨 끝의 `{token}` 부분은 사용자가 임의로 설정할 수 있는 변수이지만,

      아무나 사용할 수 없게 하기 위해 토큰을 사용하는 경우가 많다.

  - Webhook 정보 조회

    `{Telegram주소}/bot{token}/getWebhookInfo`

    새로운 메세지가 올 때마다 Webhook이 서버로 메세지를 보내므로,

    `getWebhookInfo`를 통해 알아낸 `url`을 서버에서 인식하도록 지시

    - Django의 경우 `views.py`에 `reqeust`를 받는 함수를 제작하여 json 형태로 parse

      - `urls.py`

        ```python
        from django.contrib import admin
        from django.urls import path, include
        from todos import views
        from decouple import config
        
        token = config('TOKEN')
        
        urlpatterns = [
            path('admin/', admin.site.urls),
            path('todos/', include('todos.urls')),
            path(f'{token}/', views.telegram),
        ]
        ```

      - `views.py`

        ```python
        from django.views.decorators.csrf import csrf_exempt
        # csrf token이 없다고하는 에러를 무시하기 위해 csrf_exempt 사용
        
        
        @csrf_exempt
        def telegram(request):
            
            res = json.loads(request.body)
            res.get('message').get('text')
            sendmessage(request)
            return HttpResponse('가랏')
        ```

  - Webhook 끄기

    `{Telegram주소}/bot{token}/deletewebhook`

<br><br>

## ngrok

: Ternal to Local

외부에서 local로 접속하고자 할 때, 간접적으로 발렛파킹 기능을 해줄 수 있다. 즉, 외부에서 직접적으로 local과 연결되는 것이 아니라, ngrok에게 정보를 전달하면 ngrok이 local과 소통을 한다.

- 사용 방법

  cmd >> ngork.exe 위치 >>`ngrok http 8000` + `Enter`

  8000 포트를 이용하여 외부 접속이 가능하도록 설정

- 아래 코드 블록에서는 임의로 `#` 처리한 https://7e5f168e.ngrok.io 주소를 사용하면,

  외부에서 8000 포트를 통해 local에 접속 가능

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
