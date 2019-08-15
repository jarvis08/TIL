# lotto.py
# lotto api를 통해 최신 당첨 번호 가져오기
import requests
import random


lotto_url = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo=866"
response = requests.get(lotto_url).json()
# .json :: json 파일을 파이썬 dictionary로 바꿈
winner = []
for i in range(6):
    draw = 'drwtNo{}'.format(i+1)
    winner.append(response[draw])

# 로또 번호 추천
trial = 0
while True:
    trial += 1
    ur_lotto = sorted(random.sample(range(1, 46), 6))
    count = len(set(winner) & set(ur_lotto))
    if count == 6:
        break
    else:
        print('정답 개수 = ' + count)
        pass


print(f"u tried {trial} times.")