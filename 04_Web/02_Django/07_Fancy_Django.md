# Seperate urls.py / templates / Namspaces

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

## 다른 App의 동일한 파일명의 HTML을 App별로 다르게 인식시키기

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