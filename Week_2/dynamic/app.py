from flask import Flask, render_template
import requests
import random
app = Flask(__name__)
# __content__ :: execution context
# flask 객체를 불러옴

@app.route("/")
def home():
    return render_template('home.html')
    # flask method, home.html이라고 내가 지정한 html파일을 사용


@app.route("/hello/<name>")
def hello(name):
    return render_template('hello.html', tem_name=name)
    # tem_name :: template 안에서 사용할 변수명
    # template에서는 <!-- --> 주석 사용 불가
    # html이 아닌 template의 문법으로 처리


# random으로 음식메뉴 추천 및 사진 보여주기
# dictionary 사용
@app.route('/menu')
def menu():
    food = {
        '중식':'http://mblogthumb2.phinf.naver.net/MjAxODAxMjlfMjE4/MDAxNTE3MTkyMDg1NDc4.rNgJE0gWyDuZm0NIwAbxVSXtYUL0FwKcg0jAAcTUI0kg.WI0EFy0eIQPtCTDM3rxsxwFde8lNiIOJCdfKhIRvEdUg.PNG.tjek1/%EC%A4%91%EC%8B%9D%EC%A1%B0%EB%A6%AC%EA%B8%B0%EB%8A%A5%EC%82%AC2018%2C01%2C29-105147-02.png?type=w800', 
        '양식':'http://cfile215.uf.daum.net/image/2149BC3D5246D1790FA2C7', 
        '일식':'https://t1.daumcdn.net/cfile/tistory/992132445C33770627', 
        '한식':'https://chf.or.kr/cm_data/editorImage/201403/20140303152606.gif'
        }
        # tip :: http로 시작하여 jpg로 끝나는 이미지 주소가 잘 작동
    recom = random.choice(list(food.keys()))
    return render_template('menu.html', food_name=recom, food_image=food[recom])
    # food_name :: html 파일에서 사용할 변수명
    # html 파일의 {{ food_name }}과 같은 변수의 {{ }} 안에 python 코드를 작성해도 무방

# /lotto 넘버 추천 후, 최신 로또와 비교하여 등수 알려주는 기능
@app.route('/lotto')
def lotto():
    # 우승 숫자 가져오기
    lotto_url = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo=866"
    response = requests.get(lotto_url).json()
    winner = []
    for i in range(6):
        draw = 'drwtNo{}'.format(i+1)
        winner.append(response[draw])
    # 추천 숫자 생성
    ur_lotto = sorted(random.sample(range(1, 46), 6))
    # 겹치는 개수
    count = len(set(winner) & set(ur_lotto))
    return render_template('lotto.html', str(ur_lotto)=ur_lotto, str(count)=count)

# 수정시 서버 껐다 키지 않아도 되게 하는 방법
if __name__ == "__main__":
    app.run(debug=True)