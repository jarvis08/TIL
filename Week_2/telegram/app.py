# Chatbot with Telegram
from flask import Flask, render_template, request
import requests
from pprint import pprint
from decouple import config
import random
from bs4 import BeautifulSoup

########################################################
app = Flask(__name__)
# .env 파일을 생성하고 decouple을 이용하여 token을 암호화시켰으며,
# 이는 git에 업로드 하지 않도록 .gitignore를 생성
# config(.env 속에 등록한 변수명)을 이용하여 불러옴
token = config('TELEGRAM_TOKEN')
base_url = "https://api.telegram.org"

"""getUpdates
# 아래의 webhook을 사용시 getUpdates 사용 불가
# webhook 사용하기 전까지 getUpdates method를 이용하여 공부했음
def getUpdates():
    base_url = "https://api.telegram.org"
    method = "getUpdates"
    url = f"{base_url}/{token}/{method}"
    response = requests.get(url).json()
    chat_id = response["result"][-1]["message"]["from"]["id"]
    command = response["result"][-1]["message"]["text"]
    return chat_id, command
"""
########################################################
@app.route('/')
def home():
    # chat_id, command = getUpdates()
    # return render_template('home.html', command=command, chat_id=chat_id)
    return render_template('home.html')

# telegram에 bot으로 message 보내기
@app.route('/sendMessage')
def sendMessage():
    # chat_id, command = getUpdates()
    send_to_sir = request.args.get('send_to_sir')
    method = "sendMessage"
    chat_id = "873780022"
    url = f"{base_url}/bot{token}/{method}?chat_id={chat_id}&text={send_to_sir}"
    requests.get(url)
    # return render_template('home.html', command=command, chat_id=chat_id)
    return render_template('home.html')


# 내가 챗봇에게 보낸 메세지를 그대로 돌려주는 메아리 구현
@app.route(f'/{token}', methods=['POST'])
def webhook():  
    # pprint(request.get_json())
    """
    pprint는 json을 깔끔한 형태로 보기쉽게 하기 위해 print를 대신함
    get_json을 통해 webhook이 전해준 json파일을 읽음
    그리고 이를 읽고 구조 및 인덱스를 파악하여 아래에서 text와 chat_id 추출
    """
    res = request.get_json()
    # get을 json으로 한번에 진행
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
    
    ### text 형태
    else:
        if text == 'lotto' or text == '로또':
            text = str(sorted(random.sample(range(1,46), 6)))
        elif text == '날씨':
            weather_url = "https://search.daum.net/search?w=tot&DA=YZR&t__nil_searchbox=btn&sug=&sugo=&q=%EB%82%A0%EC%94%A8"
            weather_res = requests.get(weather_url)
            weather_doc = BeautifulSoup(weather_res.text, 'html.parser')
            weather = weather_doc.select_one('#weatherColl > div.coll_cont > div > div.wrap_whole > div.cont_info > div.info_detail > div.wrap_today > a.link_weather > div > span > span.desc_temp > span').text
            temperature = weather_doc.select_one('#weatherColl > div.coll_cont > div > div.wrap_whole > div.cont_info > div.info_detail > div.wrap_today > a.link_weather > div > span > span.desc_temp > strong').text
            text = temperature + ', ' + weather
        elif '/번역' in text:
            trans_text = text.replace('/번역', '')
            papago_url = "https://openapi.naver.com/v1/papago/n2mt"
            headers = {
                'X-Naver-Client-Id' : config('NAVER_ID'),
                'X-Naver-Client-Secret' : config('NAVER_SECRET')}
            data = {
                'source' : 'ko',
                'target' : 'en',
                'text' : trans_text}
            res = requests.post(papago_url, headers=headers, data=data).json()
            text = res.get("message").get("result").get("translatedText")
    
    url = f"{base_url}/bot{token}/{method}?chat_id={chat_id}&text={text}"
    requests.get(url)
    return '', 200
    # 200을 return하여 status가 양호함을 표시
    # 정보를 요청하면 token이 들어가 있는 url로 telegram이 정보를 보내줌

if __name__ == "__main__":
    app.run(debug=True)