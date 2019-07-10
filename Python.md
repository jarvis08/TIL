# Python

---

## random

`random.sample( [], int )` : [ ] 중 int 개 만큼 sampling

`random.choice( [ ] )` : [ ] 중 1개를 선택

---

## requests

`requests.get( 'url' )` : http status code

`requests.get( 'url' ).text` : url로부터 document를 text 형태로 받음

`requests.get( 'url' ).json()` : url로부터 document를 json 형태로 받음

---

## Sort

`sorted([ ])` : list를 오름차순으로 sort

---

## webbroser

```python
import webbrowser

url = "https://search.daum.net/search?q="

keywords = ["수지", "한지민"]

for keyword in keywords:

    webbrowser.open(url + keyword)
```





```python
count = len(set(winner) & set(ur_lotto))
# winner list와 ur_lotto list를 비교할 때
# for문을 이용하는 것 보다 빠른 속도로 같은 요소의 개수를 구함
```