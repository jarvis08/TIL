# Seperate Url / Template / Namspace

---

## Project와 App의 **`urls.py` 분리**시키기

- app이 여러개 된다면 project의 urls.py에 수많은 url들이 존재
- `project > urls.py`와 `app > urls.py` 분리하여 **project의 `urls.py`는 중간 다리 역할만 하도록** 조치
- 세부 조치
- url이 `AppName/`으로 시작한다면, `include()`를 메소드를 사용하여 `AppName/urls.py`로 보내도록 작성

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

---

## 모든 app들이 **공통으로 사용할 template들을 모아서 한곳에서 관리**

- Default template 탐색 설정

  특정 Template을 찾을 때 자신(`AppName > templates`)이 갖고 있지 않다면

  다른 app의 templates에서 탐색

- 조치 내용

  App 내부가 아닌, **project directory에 template을 위치**시키고,

  모든 App이 참조할 수 있도록 조치

    1. `PROJECT/project/templates` directory 생성하여 부모 template으로 사용할 html 옮기기

    2. `PROJECT/project/settings.py`의 `TEMPLATES = [{'DIRS': []}]`에 절대경로를 추가

       `settings.py`에는 이미 BASE_DIR이 정의되어 있으므로, 이를 이용하여 경로 설정

       ```python
       # settings.py
       TEMPLATES = [
           {
               'BACKEND': 'django.template.backends.django.DjangoTemplates',
               'DIRS': [os.path.join(BASE_DIR, 'project_name','templates')],
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
       
       - `project_name` 구하기
       
         `os.path.basename(os.getcwd())`

---

## 다른 App의 동일한 파일명의 HTML을 App별로 다르게 인식시키기

- 해당 조치 이유

  Django는 **project 내부의 모든 `templates` directory들을 하나의 directory인 것 처럼 관리**

  따라서 app 별로 동일한 이름의 html 파일을 생성하면 별개로의 인식이 불가

1. `templates/AppName/` 형태로 directory를 만들어서 그 안에 위치시키기

   - 예시

     `PROJECT/project/AppName/templates/html_files`

     `PROJECT/project/AppName/templates/AppName/html_files`

     기존의 위 형태에서, 아래의 형태로 directory 구조를 변경

2. `views.py`의 `render()` 형태

   `render(request, 'AppName/template.html', context)`

   ```python
   def artii(request):
       return render(request, 'artii/artii.html', context)
   ```

---

## App 별 url namespace 구분하기

- Django에서는 url naming이 가능

  ```python
  # posts/urls.py
  urlpatterns = [
      path('posts/new/', views.index, name='posts_new'),
  ]
  ```

  ```python
  # movie/urls.py
  urlpatterns = [
      path('movie/new/', views.index, name='movie_new'),
  ]
  ```

- 하지만 여러 app들이 생겨날 경우 겹치는 이름이 발생 할 수 있으며,

  overlapping이 발생했을 경우의 문제 해결 방법이 난해

  `posts_new`, `posts_create`, `movie_new`, `movie_create`

  - url 주소 변경 또한 용이

- 따라서 app 별로 `posts`, `movie`와 같은 `app_name`을 미리 설정할 수 있도록 조치

  ```python
  # posts/urls.py
  app_name = 'posts'
  urlpatterns = [
      path('posts/new/', views.index, name='new')
  ]
  ```

  ```python
  # movie/urls.py
  app_name = 'movie'
  urlpatterns = [
      path('movie/new/', views.index, name='movie_new'),
  ]
  ```


### 활용 방법

- 기본 활용

  ```python
  # python
  app_name = 'app'
  urlpatter = [path('', views.index, 'home'),]
  return render('app:home')
  ```

  ```html
  <!-- html -->
  <a href="{% url 'app:home' %}">
  ```

- `input` parameter가 있는 경우

  ```python
  # python
  app_name = 'app'
  urlpatter = [path('<int:pk>/edit/', views.edit, 'edit'),]
  return render('app:edit', pk)
  ```

  ```html
  <!-- html -->
  <a href="{% url 'app:edit' instance.pk %}">
  ```

  

  

