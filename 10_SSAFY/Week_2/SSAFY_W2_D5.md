# SSAFY Week2 Day5

---

- Telegram chatbot API, @botfather

  1. `/newbot` : 새로운 bot 시작

  2. botfather의 지시에 따라 이름 설정

  3. https://core.telegram.org/bots/api 중 Authorizing your bot을 읽어 인증 방법 숙지

     - https://api.telegram.org/bot123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11/getMe 중

       123456~123ew11 부분을 내가 botfather에게 받은 HTTP API(token)로 교체하여 접속(getMe 기능)

       :: https://api.telegram.org/bot{token}/getUpdates

       :: user가 보낸 메세지 내역 확인

     - `sendMessage` 사용(주요 params : chat_id, text)

       1. `chat_id` : 위 url을 통해 확인한 "id"

       2. `text` : bot이 user에게 보내는 메세지

          예시 = https://api.telegram.org/bot{token}}/sendMessage?chat_id=873780022&text=왜임마

  ```python
  # Chatbot 만들기, telegram.py
  import requests
  
  # url 분해하여 사용하기
  base_url = "https://api.telegram.org/"
  token_url = config('TELEGRAM_TOKEN')
  method = ["sendMessage?"]
  chat_id = "chat_id=873780022"
  text = "text=" + "url 나누기"
  
  url = base_url + token_url + method[0] + chat_id + "&" + text
  
  response = requests.get(url)
  print(response.text)
  ```

- pprint.pprint.jason(문서명)

  `from pprint import pprint`

  `pprint(json)`

  Terminal에서 비교적 깔끔하게 json파일 구조 보여줌