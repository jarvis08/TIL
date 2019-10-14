# Make Project & Application

## Make Project

- Boiler Template, Starter Template 생성

  - Django의 Naming Convetion

    Capital Letter의 Directory 안에 같은 이름의 Lower Lettered Project가 존재

- Project 생성 방법

  ```shell
  $ mkdir PROJECT_NAME
  $ cd PROJECT_NAME
  $ django-admin startproject project_name .
  # '$ cd PROJECT_NAME' 대신에 '.' 자리에 PROJECT_NAME을 써주어도 무방
  ```
  - Start Project 생성 파일

    `PROJECT/project/__init__py`

    `PROJECT/project/settings.py`

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

<br>

<br>

## Make Application

1. app 생성

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
     
     - 변수 및 참조 활용 순서

        `Local` > `Enclosed` > `Global` > `site-packages` > `built-in`

     ```python
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

