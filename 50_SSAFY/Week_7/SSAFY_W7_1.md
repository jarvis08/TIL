# SSAFY Week7 Day1

---

```
190809_출결확인서_서울1반_조동빈
- 대분류: 공가(취업) 
- 일시: 2019.08.09 (금)
- 회사명: 가나다 회사
- 내용: 가나다 회사 수시 채용 면접에 참석하여 결석 소명 신청합니다.
```

---

- Django의 주장

  - Versatile, 다용도의
    Complete, 완결성있는
    Secure, 안전한
    Scalable, 확장성있는
    Maintainable, 쉬운유지보수
    Portable, 포터블한

- Django의 성격

  Opinionated, 독선적
  django에서 문법을 지정하여, 초기 문법 공부가 요구됨

  - Unopinionated, 관용적

    많은 것들을 허용하지만, 차후 많은 것들이 제한되거나 충돌

- Youtube, Instagram, Mozilla, NASA 등에서 사용하여 Web App 제작

- <https://hotframeworks.com/>

  framework 순위 확인 가능

- Static Web

  누구에게나 같은 contents를 보여주는 page

- Dynamic Web, Web Application Program(Web App)

  유저 별, 상황 별 다른 페이지를 return

- Framework

  기본적인 구조나 필요한 코드들을 알아서 제공

  웹 서비스를 만드는데에 집중 가능

- Flask

  ```python
  @app.route('url')
  def index():
      return render_template
  ```

  page 별로 위의 패턴을 적용하여 작성해야함

  e.g., post, posts, posts/edit, posts/delete.... 기능 별로 모두 제작하는 **Micro Framework**

  - Maintenance 또한 복잡해짐

- **MVC**, Django는 **MTV**

  - Model

    데이터를 관리하며, Database로 접근

  - Template

    사용자가 보는 화면(HTML)을 관리

    - Flask의 `render_template()` 역할

  - **View**

    Model과 Template의 **중간 관리자**

    - URLS를 통해 `url`을 받고, 이를 Model 혹은 Template으로 전송하여 작업을 진행

---

- Boiler Template, Starter Template 생성

  - Django의 Naming Convetion

    Capital Letter의 Directory 안에 같은 이름의 Lower Lettered Project가 존재

  ```shell
  $ mkdir PROJECT
  $ cd PROJECT
  $ django-admin startproject project .
  ```

  - 생성 파일

    `PROJECT/project/__init__py`

    `PROJECT/project/wettings.py`

    `PROJECT/project/urls.py`

    `PROJECT/project/wsgipy`

    `PROJECT/manage.py`

    - `urls.py`

      Flask의 `@app.route('posts')`의 기능을 수행하는, url을 다루는 문지기 역할

      `urls.py`의 `urlpatterns`에 `path()`를 추가

- 서버 실행

  ```shell
  $ python manage.py runserver
  ```

  Webbrowser `localhost:8000` 접속하여 확인

- Django는 `project` 안에 여러 `app`들을 생성하여 로직을 구현

  e.g., app1 - 게시판, app2 - 회원관리...

---

## app 만들기

1. app 생성하기

    ```shell
    $ python manage.py startapp NewApp
    ```

    `/앱이름/` directory를 생성하며, 구조는 다음과 같다.

    - `PROJECT/NewApp/migrations/__init__.py`

      `PROJECT/NewApp/__init__.py`

      `PROJECT/NewApp/admin.py`

      `PROJECT/NewApp/apps.py`

      `PROJECT/NewApp/models.py`

      `PROJECT/NewApp/tests.py`

      `PROJECT/NewApp/views.py`

2. `PROJECT/project/urls.py`에 NewApp pattern 추가

     `path( url경로/, view함수)`
     
     ```python
     from NewApp import views
     urlpatterns = [
         path('admin/', admin.site.urls),
         path('index/', views.index),
     ]
     ```
     
     - url 끝에 `/`를 붙여주는게 Django의 Convention
     - 마지막 요소일 지라도 `,`를 붙여주는 것이 Django의 Convention
     
3. `PROJECT/project/settings.py`에 app 설치를 기록

     ```python
     INSTALLED_APPS = [
         'NewApp',
         'django.contrib.admin',
         'django.contrib.auth',
         'django.contrib.contenttypes',
         'django.contrib.sessions',
         'django.contrib.messages',
         'django.contrib.staticfiles',
     ]
     ```

     항상 목록의 맨 위에 써주는 것이 Convention

4. `PROJECT/NewApp/view.py`에 함수 작성

   ```python
   from django.shortcuts import render
   
   # Create your views here.
   def index(request):
       return render(request, 'index.html')
   ```

   - `request`는 사용자의 input을 받음

   - `return` 없이 `pass`로 설정할 경우 `index/` 경로 접속 시 `HttpResponse`가 존재하지 않는다고 `ValueError` 발생

     ```python
     from django.shortcuts import render
     
     # Create your views here.
     def index(request):
         return render(request, 'index.html')
     
     def home(request):
         pass
     ```

   - `HttpResponse` class 사용하기

     - 확인 방법

       1. Django Github

       2. local pip, `site-packages/`

          `Local` - `Enclosed` -  `Global` - `site-packages` - `built-in`

     ```python
     from django.shortcuts import render
     from django.http import HttpResponse
     
     # Create your views here.
     def index(request):
         return render(request, 'index.html')
     
     def home(request):
         return HttpResponse("Hello Django!")
     ```

     - `HttpResponse`를 통해 String을 바로 전달하는 것이 **가능**은 하다.

       **`But, 가능하다 !=사용하자`**

     - `render()`는 `HttpResponse`를 생성하는 함수임을 확인 가능

   - html로 변수 전달하기

     ```python
     def home(request):
         name = '조동빈'
         data = ['강동주', '김지수', '정의진']
         context = {
             'myname': name,
             'class': data,
         }
         # flask는 jinja를 사용하며
         # django는 DTL이라는 template engine을 사용하며, dict 형태로 전달
         # 아래 두가지가 모두 가능하지만 context라고 넣어 보내는 것으로 하자.
         return render(request, 'home.html', 'myname': name, 'class': data)
         return render(request, 'home.html', context)
     ```

     ```html
     <body>
       <h1>데이터를 넘겨 받는 법</h1>
       <p>{{ myname }}</p>
       <p>{{ class }}</p>
       {% for item in class %}
         <p>{{ item }}</p>
       {% endfor %}
     </body>
     ```

     ```
     # 결과
     데이터를 넘겨 받는 법
     조동빈
     ['강동주', '김지수', '정의진']
     강동주
     김지수
     정의진
     ```

5. `PROJECT/NewApp/templates` directory 생성

6. `PROJECT/NewApp/templates/index.html` 생성

---

## DTL, Django Template Language

- `views.py` 함수의 parameter로 항상 **`request`**를 기입

- 마지막 요소일 지라도 `,`를 붙여주기

- `render(request, url, 변수Dict)` 형태로 사용하며,

  `render()` 함수는 `HttpResponse`를 생성하여 반환

  e.g., `render(request, 'home.html', context)`

  - 변수들을 묶어서 `context`라는 dictionary 변수 안에 포함시켜 한번에 `render()`의 인수로 기입

  - `context`라는 단어는 무엇을 사용하든 무관하지만,

    context를 넘긴다는 의미 하에 관례적으로 사용되며, `ctx`로 줄여서 표현

    ```python
    def home(request):
        name = '조동빈'
        data = ['강동주', '김지수', '정의진']
        context = {
            'myname': name,
            'class': data,
        }
        return render(request, 'home.html', context)
    ```

- url은 항상 **`/`**로 종료

- `for` (2중도 가능)

  ```html
  {% for s in lotto %}
  	<p>{{ s }}</p>
  {% endfor %}
  ```

  - 숫자 counting하며 출력하기

    `{{ forloop.counter }}`

    ```html
    {% for movie in movie_dataset %}
      <p>{{ forloop.counter }}. {{ movie }}</p>
    {% endfor %}
    ```

  - 데이터가 없을 때

    ```html
    {% for movie in empty_dataset %}
      <p>{{ movie }}</p>
    {% empty %}
      <p>영화 데이터가 없습니다.</p>
    {% endfor %}
    ```

- `if`

  ```html
  {% if True %}
    <p>이건 참일때</p>
  {% endif %}
  ```

  ```html
  {% for s in lotto %}
    {% if s > 40 %}
      <p class="nums b4">{{ s }}</p>
    {% elif s > 30 %}
      <p class="nums b3">{{ s }}</p>
    {% elif s > 20 %}
      <p class="nums b2">{{ s }}</p>
    {% elif s > 10 %}
      <p class="nums b1">{{ s }}</p>
    {% else %}
      <p class="nums b0">{{ s }}</p>
    {% endif %}
  {% endfor %}
  ```

- `helper` / `filter`

  - `helper` example

      ```html
      {% lorem %}
      ```

      - lorem 활용하기

          ```html
          {% lorem [count] [method] [random] %
          ```

          <https://docs.djangoproject.com/en/2.2/ref/templates/builtins/>

          ```
          {% lorem 3 p %}
          ```
      
  - `filter`

    `{{ html변수|filter함수 }}`
    
    - `length`
    
      길이 표시
    
    - `truncatechars:value`
    
      글자수를 `value`만큼 제한하여, 나머지는 `...` 처리
    
        ```html
      {% for movie in empty_dataset %}
        <p>{{ movie|length }} {{ movie|truncatechars:5 }}</p>
      {% endfor %}
        ```
    
    - `datetime`
    
      ```html
      <p>{% now 'd.m.y' %}</p>
      <p>{% now 'd M Y' %}</p>
      <p>{% now 'Y년m월d일 h시i분 a D' %}</p>
      ## 출력
      12.08.19
      12 Aug 2019
      2019년08월12일 03시57분 p.m. Mon
      ```
    
      - `settings.py` 에서 Time Zone을 변경해 주어야 원하는 장소의 시간대
    
        ```python
        # TIME_ZONE = 'UTC'
        TIME_ZONE = 'Asia/Seoul'
        ```
    
    - int 숫자 처리
    
      ```html
      {{ num|add:10}}
      ## 출력
      20
      ```
    
      But `가능 != 해도된다`

- `extends`

  Template 상속하여 편의성 급증

  e.g., Bootstrap을 Template마다 적용시키기

- `block`

---

## App에 기능 추가하기

- 이미 존재하는 App에 기능 추가하기

  1. `urls.py`에 `url`과  `view 함수 위치` 추가
  2. `views.py`에 함수 작성
  3. 함수에 대한 html파일 제작

- 숫자 변수 받기

  - `urls.py`

    ```python
    urlpatterns = [
        # index/, home/ 이라는 주문이 들어올 시, views를 실행
        path('admin/', admin.site.urls),
        path('cube/<int:num>/', views.cube),
    ]
    ```

  - `views.py`

    ```python
    def cube(request, num):
        result = num ** 3
        context = {
            'result': result,
        }
        return render(request, 'cube.html', context)
    ```

- **`<input>` 변수 전달하기**

  - `index.html`의 **`GET`** 방식의 `<form>`

    ```html
    <form action="/match/" method="GET">
      당신의 이름 : <input type="text" name="me">
      당신이 좋아하는 분의 이름 : <input type="text" name="you">
      <button type="submit">제출</button>
    </form>
    ```

      ```python
    def match(request):
        goonghap = random.randint(50, 100)
        me = request.GET.get('me')
        you = request.GET.get('you')
        path_1 = request.path_info
        path_2 = request.path
        scheme = request.scheme
        method = request.method
        host = request.get_host
        context = {
            'me': me,
            'you': you,
            'goonghap': goonghap,
            'path_1': path_1,
            'path_2': path_2,
            'scheme': scheme,
            'method': method,
            'host': host,        
        }
        return render(request, 'match.html', context)
      ```
  
    ```html
  <!-- match.html -->
    <h1> {{ me }}님과 {{ you }}님의 궁합은 {{ goonghap }}%입니다.</h1>
    ```
  <p>{{ path_1 }}</p>
    <p>{{ path_2 }}</p>
    <p>{{ scheme }}</p>
    <p>{{ method }}</p>
    <p>{{ host }}</p>   
    ```
    
    ```
    ## result
    조동빈님과 박현지님의 궁합은 51%입니다.
    /match/
    /match/
    http
    GET
    localhost:8000
    ```
    
  - `index.html`의 **`POST`** 방식의 `<form>`
  
    `POST`의 경우 `CSRF Token`으로 넣어버리는 것과, 무시하는 방법이 존재
  
    Token으로 넣기 - `{% csrf_token %}`
  
    ```html
    <!-- index.html -->
    <form action="/match/" method="POST">
      {% csrf_token %}
      당신의 이름 : <input type="text" name="me">
      당신이 좋아하는 분의 이름 : <input type="text" name="you">
      <button type="submit">제출</button>
    </form>
    ```
  
    ```python
    def match(request):
        goonghap = random.randint(50, 100)
        me = request.POST.get('me')
        you = request.POST.get('you')
        method = request.method
        context = {
            'me': me,
            'you': you,
            'goonghap': goonghap,
            'method': method,
        }
        return render(request, 'match.html', context)
    ```
  
    ```html
    <!-- match.html -->
    <h1> {{ me }}님과 {{ you }}님의 궁합은 {{ goonghap }}%입니다.</h1>
    <p>{{ method }}</p>
    ```
  
    ```
    ## result
    조동빈님과 박현지님의 궁합은 51%입니다.
    POST
    ```
  
    
  
    
  
  
  
  