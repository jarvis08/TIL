# API

## Papago NMT API

```python
# Papago NMT API
# guide
"""
curl "https://openapi.naver.com/v1/papago/n2mt" \
-H "Content-Type: application/x-www-form-urlencoded; charset=UTF-8" \
-H "X-Naver-Client-Id: RfwcGLCq6s9wOnT09ZMF" \
-H "X-Naver-Client-Secret: h6OJ2b_Das" \
-d "source=ko&target=en&text=만나서 반갑습니다." -v
"""
################## python 적용 방법
papago_url = "https://openapi.naver.com/v1/papago/n2mt"
headers = {
    'X-Naver-Client-Id' : config('NAVER_ID'),
    'X-Naver-Client-Secret' : config('NAVER_SECRET')}
data = {
    'source' : 'ko',
    'target' : 'en',
    'text' : '안녕하세요, 번역기입니다'}
res = requests.post(papago_url, headers=headers, data=data).json()
text = res.get("message").get("result").get("translatedText")
print(text)
```

<br>

<br>

## Clova Face Recognition

https://developers.naver.com/docs/clova/api/CFR/API_Guide.md?origin_team=TL97JQ9KQ

```python
@app.route(f'/{token}', methods=['POST'])
def webhook():  
    res = request.get_json()
    # 전달받은 내용 dic으로 변환
    text = res.get('message').get('text')
    chat_id = res.get('message').get('chat').get('id')
    method = "sendMessage"
    pprint(res)
    
    ### image 형태
    # if res.get("message").get("photo") is not None: 동일한 조건문
    if res.get("message").get("photo")[-1].get("file_id"):
        # file_id를 통해 사진 파일이 저장된 PATH를 알아내는 과정
        file_id = res.get('message').get('photo')[-1].get('file_id')
        file_res = requests.get(f"{base_url}/bot{token}/getFile?file_id={file_id}")
        file_path = file_res.json().get("result").get('file_path')
        file_url = f"{base_url}/file/bot{token}/{file_path}"
        image = requests.get(file_url, stream=True)

        # clova api 연결
        clova_url = "https://openapi.naver.com/v1/vision/celebrity"
        headers = {
            'X-Naver-Client-Id': config("NAVER_ID"),
            'X-Naver-Client-Secret': config("NAVER_SECRET")}
        files = {'image': image.raw.read()}
        clova_res = requests.post(clova_url, headers=headers, files=files)
        text = clova_res.json().get('faces')[0].get('celebrity').get('value')
       url = f"{base_url}/bot{token}/{method}?chat_id={chat_id}&text={text}"
    
    requests.get(url)
    return '', 200
    # 200을 return하여 status가 양호함을 표시
    # 정보를 요청하면 token이 들어가 있는 url로 telegram이 정보를 보내줌
```

<br><br>

## Telegram chatbot API, @botfather

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
# Chatbot 만들기, .py
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

