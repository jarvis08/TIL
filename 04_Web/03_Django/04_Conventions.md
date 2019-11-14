# Django's Conventions

- New Project 생성 시

  Capital Letter의 Directory 안에 같은 이름의 Lower Lettered Project가 존재

  ```
  NEWPROJECT/newproject/
  NEWPROJECT/manage.py
  ```

<br>

- urls의 `path(url, vies.def)` 에서 url과 def는 동일한 이름으로 하는 것이 Convention

  ```python
  urlpatterns ={
      path('admin/', admin.site.urls),
      path('result/', views.result),
      path('index/', views.index),
  }
  ```

- url은 항상 **`/`**로 종료

- Root 경로 설정하기

  주소 `localhost:8000` 접속 시의 화면

  ```python
  urlpatterns = [
      # root page는 '' 빈 url을 부여
      path('', views.index),
      path('admin/', admin.site.urls),
  ]
  ```

- 마지막 요소일 지라도 `,`를 붙여주기

<br>

- `views.py` 함수의 parameter로 항상 **`request`**를 기입

<br>

- Inheritance 설정 시 `block` 이름은 **`body`** 혹은 **`content`**

  ```html
  <!-- 부모 html 내부 -->
  {% block body %}
  {% endblock %}
  ```

  