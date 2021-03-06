# Tips - IPython

## IPython, 디버깅 편하게 하기

**IPython**은 **Python Interactive shell**을 강화한 형태이며, IPython을 _웹 페이지 형식_으로 심화한건 **IPython Notebook**이다.

IPython의 `embed()`를 활용하면 편리하게 디버깅할 수 있다.

<br>

### 설치 및 코드 설정

`pip install ipython[all]`

```python
# settings.py
INSTALLED_APPS = [
    'django_extensions',]
```

```python
# views.py
from IPython import embed

def create(request):
    if request.method == 'POST':
        form = ArticleForm(request.POST)
        # 작업을 중단하고 ipython shell을 킨다.
        embed()
        #######

        if form.is_valid():
            title = form.cleaned_data.get('title')
            content = form.cleaned_data.get('content')
            article = Article.objects.create(title=title, content=content)
            return redirect(article)
        else:
            return redirect('articles:create')
    ...
```

<br>

### 사용하기

1. 서버 실행
2. create 작업 시작
3. 제출
4. **embed()가 call 되기 이전의 작업들이 진행된 상태**로 Ipython shell이 실행됨

```bash
[15/Oct/2019 09:42:04] "GET / HTTP/1.1" 404 2029
[15/Oct/2019 09:42:10] "GET /articles/ HTTP/1.1" 200 1739
[15/Oct/2019 09:42:12] "GET /articles/create/ HTTP/1.1" 200 1722
##############################################################
Python 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 19:29:22) [MSC v.1916 32 bit (Intel)]
Type 'copyright', 'credits' or 'license' for more information
IPython 7.8.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: form
Out[1]: <ArticleForm bound=True, valid=Unknown, fields=(title;content)>

In [2]: request.POST
Out[2]: <QueryDict: {'csrfmiddlewaretoken': ['NN6Egy8gUcEeEUYpoqE3nDKET7ag0ftcH0EtzY2ziool8zRNaVxRMSS594972MAP'], 'title': ['짱짱맨'], 'content': ['짜앙']}>

In [3]: request.POST.get('title')
Out[3]: '짱짱맨'

# 토큰 값 확인, csrf middleware token
In [4]: request.POST.get('csrfmiddlewaretoken')
Out[4]: 'NN6Egy8gUcEeEUYpoqE3nDKET7ag0ftcH0EtzY2ziool8zRNaVxRMSS594972MAP'

# 우리가 정의한 forms.py의 ArticleForm 클래스의 객체임을 확인 가능
In [5]: type(form)
Out[5]: articles.forms.ArticleForm

# Validation 확인
In [6]: form.is_valid()
Out[6]: True

# Validation을 통과한, 클린한 데이터의 목록을 딕셔너리 형태로 저장
# 통과하는 데이터가 없다면 {} 형태의 빈 딕셔너리
In [7]: form.cleaned_data
Out[7]: {'title': '짱짱맨', 'content': '짜앙'}

In [8]: exit
##############################################################
[15/Oct/2019 09:53:39] "POST /articles/create/ HTTP/1.1" 302 0
[15/Oct/2019 09:53:40] "GET /articles/8/ HTTP/1.1" 200 1233
```

```bash
In [2]: form.as_table()
Out[2]: '<tr><th><label for="id_title">Title:</label></th><td><input type="text" name="title" value="테스또" maxlength="50"
required id="id_title"></td></tr>\n<tr><th><label for="id_content">Content:</label></th><td><textarea name="content" cols="40" rows="10" required id="id_content">\n떼스트</textarea></td></tr>'

In [3]: form.as_p()
Out[3]: '<p><label for="id_title">Title:</label> <input type="text" name="title" value="테스또" maxlength="50" required id="id_title"></p>\n<p><label for="id_content">Content:</label> <textarea name="content" cols="40" rows="10" required id="id_content">\n떼스트</textarea></p>'

In [4]: form.as_ul()
Out[4]: '<li><label for="id_title">Title:</label> <input type="text" name="title" value="테스또" maxlength="50" required id="id_title"></li>\n<li><label for="id_content">Content:</label> <textarea name="content" cols="40" rows="10" required id="id_content">\n떼스트</textarea></li>'
```


