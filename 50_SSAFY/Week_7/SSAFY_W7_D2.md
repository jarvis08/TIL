# SSAFY Week7 Day2

---

- 결제 모듈 붙이기

  아임포트, iamport

  <https://www.iamport.kr/>

- wishket, freemoa 등 외주 사이트

---

## Root URL

- Root 경로 설정하기

  주소 `localhost:8000` 접속 시의 화면

  ```python
  urlpatterns = [
      # root page는 '' 빈 url을 부여
      path('', views.index),
      path('admin/', admin.site.urls),
  ]
  ```

---

- urls의 `path(url, vies.def)` 에서 url과 def는 동일한 이름으로 하는 것이 Convention

  ```python
  urlpatterns ={
      path('admin/', admin.site.urls),
      path('result/', views.result),
      path('index/', views.index),
  }
  ```

---

## Django Template Inheritance

- DRY, Do not Repeat Yourself

- Navbar라는 요소는 모든 페이지에 있어야 하므로 다른 페이지에서도 가능해야 한다.

  따라서 이를 복사&붙여넣기 보다는 상속을 이용하여 진행

  1. 공통적으로 사용할 템플릿(코드)을 추출

  2. 해당 템플릿(코드)를 파일로 따로 생성

     `FIRST_APP/first_app/templates/base.html`

     - **body 끝부분에 block을 설정**

       코드 구멍을 뚫는 역할이며, **상속받는 페이지의 내용들이 들어갈 곳**을 정의

     ```html
     <!-- base.html -->
     <!-- inherite 할 html 내용들 -->
     <!-- body는 내가 설정하는 이름이며, 관례로 body 혹은 content -->
       {% block body %}
       {% endblock %}
     </body>
     ```

  3. 활용할 다른 템플릿 파일에서 불러와서 사용

     **상속할 template(base.html)과 중복되는 내용은 모두 삭제**

     - `base.html`을 상속받는다는 코드를 가장 위에 작성

       ```html
       {% extends 'base.html' %}
       ```

     - `base.html`에서 설정한 block에 들어갈 내용들(상속 받지 않는)을 작성

       ```html
       {% block body %}
       
         <h2>For문</h2>
         <p>{{ myname }}</p>
         <p>{{ class }}</p>
         {% for item in class %}
           <p>{{ item }}</p>
         {% endfor %}
       
       {% endblock %}
       ```

     - `home.html` 전체 코드

       ```html
       <!-- home.html -->
       <!-- inherite 받는 html -->
       {% extends 'base.html' %}
       {% block body %}
       
         <h2>For문</h2>
         <p>{{ myname }}</p>
         <p>{{ class }}</p>
         {% for item in class %}
           <p>{{ item }}</p>
         {% endfor %}
       
       {% endblock %}
       ```

---

## Partial View, Rendering, Template

- Partial Temlplate

  **파일명** 앞에 **`_`**를 붙이는 것이 관례

  - 예시 파일명

    `_footer.html`

    `_nav.html`

- `{% extends '_nav.html' %}` 대신 **Partial Rendering**인 **`inlude`**를 사용

  `extends`의 경우 `_nav.html`을 메인으로 하며,

  `include`의 경우 본 html을 메인으로 하여 `_nav.html`을 첨부하여 활용

  ```html
  {% include '_nav.html' %}
  ```

- Partial Template

  ```html
  <!-- _nav.html -->
  <nav class="navbar navbar-expand-lg navbar-light bg-light sticky-top">
      <a class="navbar-brand" href="#">잡동사니</a>
      <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav"
        aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarNav">
        <ul class="navbar-nav">
          <li class="nav-item active">
            <a class="nav-link" href="#">Home <span class="sr-only">(current)</span></a>
          </li>
          <li class="nav-item">
            <a class="nav-link" href="/cube/">세제곱계산기</a>
          </li>
          <li class="nav-item">
            <a class="nav-link" href="/lotto/">로또</a>
          </li>
          <li class="nav-item">
            <a class="nav-link" href="/home/">DTL 정리</a>
          </li>
        </ul>
      </div>
    </nav>
  ```

  ```html
  <!-- _footer.html -->
  <footer class="d-flex justify-content-center fixed-bottom bg-dark text-white">
    <p class="mb-0">Copyright. Dongbin Cho</p>
  </footer>
  ```

  ```html
  <!-- base.html -->
  <!DOCTYPE html>
  <html lang="en">
  
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
      integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <title>Document</title>
  </head>
  
  <body>
    {% include '_nav.html' %}
  
    {% block body %}
    {% endblock %}
    
    {% include '_footer.html' %}
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js"
      integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous">
    </script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"
      integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous">
    </script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"
      integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous">
    </script>
  </body>
  </html>
  ```

---

## artii API 사용해보기

- 사용자 입력 받기

  /artii/

- artii API를 통해 ascii art를 보여주는 앱

  /artii/result/

  ```python
  # urls.py
  from pages import views
  from artii import views as artii_views
  
  urlpatterns = [
      # index/, home/ 이라는 주문이 들어올 시, views를 실행
      path('', views.index),
      path('admin/', admin.site.urls),
      path('index/', views.index),
      path('home/', views.home),
      path('lotto/', views.lotto),
      path('cube/<int:num>/', views.cube),
      path('match/', views.match),
      path('artii/', artii_views.artii),
      path('artii/result/', artii_views.artii_result),
  ]
  ```

  ```python
  # settings.py
  INSTALLED_APPS = [
      'artii',
      'pages',
      'django.contrib.admin',
      'django.contrib.auth',
      'django.contrib.contenttypes',
      'django.contrib.sessions',
      'django.contrib.messages',
      'django.contrib.staticfiles',
  ]
  ```

  ```python
  # views.py
  from django.shortcuts import render
  import requests
  
  def artii(request):
      font_url = 'http://artii.herokuapp.com/fonts_list'
      response = requests.get(font_url).text
      font_list = response.split()
      context = {
          'font_list': font_list,
      }
      return render(request, 'artii.html', context)
  
  def artii_result(request):
      string = request.GET.get('string')
      font = request.GET.get('font')
      url = f'http://artii.herokuapp.com/make?text={string}&font={font}'
      # url = f'http://artii.herokuapp.com/make?text={string}'
      res = requests.get(url).text
      context = {
          'artii_result': res,
      }
      return render(request, 'artii_result.html', context)
  ```

  ```html
  <!-- artii.html -->
  <body>
    <h1>아스키 아트 제조기</h1>
    <form action="/artii/result/" method="GET">
      <input type="text" name="string">
      <select name="font" id="">
        {% for font in font_list %}
        <option>{{ font }}</option>
        {% endfor %}
      </select>
      <button type="submit">submit</button>
    </form>
  </body>
  ```

  ```html
  <!-- artii_result.html -->
  <body>
  <pre>{{ artii_result }}</pre>
  
  </body>
  ```

- `<pre>` 조작하기 전에 text를 그대로 출력

  ---

## 장고스럽게 제작하기

- Project와 App의 `urls.py` 분리시키기

  app이 여러개 된다면 project의 urls.py에 수많은 url들이 존재

  `project > urls.py`와 `app > urls.py` 분리하여 project는 중간 다리 역할만 하도록

  - url이 `AppName/`으로 시작한다면 `AppName/urls.py`로 보내라

    ```python
    # PROJECT/project/urls.py
    from django.contrib import admin
    from django.urls import path, include
    
    urlpatterns = [
        path('', views.index),
        path('admin/', admin.site.urls),
        # path('AppName/', views.AppName),
        # path('AppName/FuncName/', views.FuncName_result),
        path('AppName/', include('AppName.urls')),
    ]
    ```

    ```python
    # PROJECT/AppName/urls.py
    from django.urls import path, include
    from . import views
    
    urlpatterns = [
        # path('AppName/', views.AppName),
        path('', views.FuncName),
        # path('AppName/FuncName/', views.FuncName_result),
        path('FuncName/', views.FuncName_result),
    ]
    ```

- 모든 app들의 공통 template 모아서 한곳에서 관리하기

  - Default template 탐색 설정

    특정 Template을 찾을 때엔 자신(`AppName > templates`)이 갖고 있지 않다면,

    다른 app의 templates에서 탐색

  1. `PROJECT/project/templates` directory 생성하여 부모 Template으로 사용할 html 옮기기

  2. `PROJECT/project/settings.py`의 `TEMPLATES = [{'DIRS': []}]`에 절대경로를 추가

     `settings.py`에는 이미 BASE_DIR이 정의되어 있으므로, 이를 이용하여 경로 설정

     ```python
     # settings.py
     TEMPLATES = [
         {
             'BACKEND': 'django.template.backends.django.DjangoTemplates',
             'DIRS': [os.path.join(BASE_DIR, 'first_app','templates')],
             'APP_DIRS': True,
             'OPTIONS': {
                 'context_processors': [
                     'django.template.context_processors.debug',
                     'django.template.context_processors.request',
                     'django.contrib.auth.context_processors.auth',
                     'django.contrib.messages.context_processors.messages',
                 ],
             },
         },
     ]
     ```

---

## 동일한 html 파일명을 app별로 다르게 인식하기

1. `templates/AppName/` 형태로 directory를 만들어서 그 안에 위치시키기

   `PROJECT/project/AppName/templates/html_files`

   에서

   `PROJECT/project/AppName/templates/AppName/html_files`

   구조로 변경

2. `views.py`의 `render()` 형태

   `render(request, 'AppName/template.html', context)`

   ```python
   def artii(request):
       return render(request, 'artii/artii.html', context)
   ```

---

## os Path 조작하기

- OS마다 Path 정의 방식이 다르지만,

  python이 알아서 path를 찾을 수 있도록 조치

- `os.getcwd()`

  current working directory

- `os.path.join(a, b, c)`

  `a/b/c` 형태로 `a` 경로에 `b`와 `c`의 경로를 추가

  ```python
  import os
  current = os.getcwd()
  templates_path = os.path.join(current, 'Directory_name')
  ```

  

