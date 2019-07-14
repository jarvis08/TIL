import requests
import bs4

# 1. 요청 보내기
op_gg = 'https://www.op.gg/summoner/userName=원빈우빈동빈'
response = requests.get(op_gg)
# 2. html 받기
document = bs4.BeautifulSoup(response.text, 'html.parser')
# 3. 정보 parsing
win = document.select_one('#SummonerLayoutContent > div.tabItem.Content.SummonerLayoutContent.summonerLayout-summary > div.SideContent > div.TierBox.Box > div > div.TierRankInfo > div.TierInfo > span.WinLose > span.wins').text
lose = document.select_one('#SummonerLayoutContent > div.tabItem.Content.SummonerLayoutContent.summonerLayout-summary > div.SideContent > div.TierBox.Box > div > div.TierRankInfo > div.TierInfo > span.WinLose > span.losses').text

print(win[:-1] + ' 승')
print(lose[:-1] + ' 패')