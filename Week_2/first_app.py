from flask import Flask
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime
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

# cube : 세제곱
# ex) /cube/2 == 8
@app.route("/cube/<num>")
def cube(num):
    integer = int(num) ** 3
    return str(integer)
    # return 값은 언제나 string || tuple || dictionary

# 로또 번호 추천
@app.route("/lotto")
def lotto():
    recommended = sorted(random.sample(range(1,46), 6))
    return str(recommended).replace('[', '').replace(']', '')

# 점심 메뉴 추천
@app.route("/menu")
def menu():
    menu = ['피자', '장어구이', '삼겹살', '고등어구이']
    choice = random.choice(menu)
    return choice

# Kospi
@app.route("/kospi")
def kospi():
    finance_url = "https://finance.naver.com/sise/"
    response = requests.get(finance_url).text
    document = BeautifulSoup(response, 'html.parser')
    kospi = document.select_one('#KOSPI_now').text
    return kospi

# 새해?
@app.route("/newyear")
def newyear():
    month = datetime.now().month
    day = datetime.now().day
    year = datetime.now().year
    if month == 1 and day == 1:
        return '<h1>YEEEEEEES!!!</h1>'
    else:
        return '<h1>NOOOOOOPE!!!</h1><br>Today is {}.{}.{}'.format(day, month, year)

# /index
@app.route("/index")
def index():
    return "<html><head></head><body><h1>홈페이지</h1><p>이건내용</p></body></html>"