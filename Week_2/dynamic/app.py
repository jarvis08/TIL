from flask import Flask, render_template, request
from faker import Faker
import random
import requests
import bs4
fake = Faker('ko_KR')
app = Flask(__name__)

@app.route('/')
def home():
    google_img = 'http://pngimg.com/uploads/google/google_PNG19642.png'
    naver_img = 'https://logoproject.naver.com/download/NAVER_Logo.jpg'
    daum_img = 'data:image/png;base64,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'
    return render_template('day_4_home.html', google_logo=google_img, naver_logo=naver_img, daum_logo=daum_img)


@app.route('/pastlife')
def pastlife():
    return render_template('pastlife.html')

existance = {}
@app.route('/result')
def result():
    input_name = request.args.get('name')
    if input_name in existance:
        # True, False로 응답
        fake_job = existance[input_name]
    else:
        fake_job = fake.job()
        existance[input_name] = fake_job
    # flask는 사용자의 요청을 request 안에 담아두고 있다.
    return render_template('result.html', ur_name=input_name, ur_job=fake_job)

@app.route('/goonghap')
def goonghap():
    return render_template('goonghap.html')

goong = {}
@app.route('/goonged')
def goonged():
    you = request.args.get('babo')
    another = request.args.get('another')
    
    # dic 속의 dic 목록에 있는지 확인
    # {'조동빈': {'강동주': 12, '김병철': 1}, '이여진': {...}}
    if you in goong:
        if another in goong[you]:
            per = goong[you][another]
        else:
            per = random.randint(0, 101)
            goong[you][another] = per
    else:
        per = random.choice(range(0, 101))
        goong[you] = {another : per}

    """ Tuple 이이용
    if (you, another) in goong:
        per = goong[(you, another)]
    else:
        per = random.choice(range(0, 101))
        goong[(you, another)] = per
    """
    # percentage에 따라 다른 사진
    if per > 50:
        image = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRILdBZ4RX4JVuqulCHhOu5ziNilA42rb0GiD2g-ZJWyiSINgEX"
    else:
        image = "https://pbs.twimg.com/profile_images/560849466409115648/Ff9Ppsgd.jpeg"
    return render_template('goonged.html', you=you, another=another, per=per, image=image)

# goong에 있는 사람 모두 출력하기

@app.route('/admin')
def admin():
    babong = ''
    for k,v in goong.items():
        for a in v:
            babong = babong + f'{k} 하쮸 {a}\n'
    return babong


# op.gg 전적 검색기 유사 사이트
@app.route('/opgg')
def opgg():
    return render_template('opgg.html')

@app.route('/opgg_result')
def opgg_result():
    user_id = request.args.get('user_id')
    url = 'https://www.op.gg/summoner/userName={}'.format(user_id)
    response = requests.get(url)
    document = bs4.BeautifulSoup(response.text, 'html.parser')

    win = document.select_one('#SummonerLayoutContent > div.tabItem.Content.SummonerLayoutContent.summonerLayout-summary > div.SideContent > div.TierBox.Box > div > div.TierRankInfo > div.TierInfo > span.WinLose > span.wins').text
    lose = document.select_one('#SummonerLayoutContent > div.tabItem.Content.SummonerLayoutContent.summonerLayout-summary > div.SideContent > div.TierBox.Box > div > div.TierRankInfo > div.TierInfo > span.WinLose > span.losses').text
    most_pick = document.select_one('#GameAverageStatsBox-summary > div.Box > table > tbody > tr:nth-child(1) > td.MostChampion > ul > li:nth-child(1) > div.Content > div.Name')
    pubg_rating = document.select_one('#rankedStatsWrap > div.ranked-stats-wrapper__list > div:nth-child(1) > div > div:nth-child(3) > div > div > div > div > div.ranked-stats__layout.ranked-stats__layout--rank > div > div:nth-child(2) > div.ranked-stats__rating-point')

    return render_template('opgg_result.html', user_id=user_id, win=win, lose=lose, most_pic=most_pick, pubg_rating=pubg_rating)


if __name__ == '__main__':
    app.run(debug=True)