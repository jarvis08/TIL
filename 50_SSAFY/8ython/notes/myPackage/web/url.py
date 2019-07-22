def make_url(key, targetDt):
    base = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json'
    return f'{base}?key={key}&targetDt={targetDt}' 
 