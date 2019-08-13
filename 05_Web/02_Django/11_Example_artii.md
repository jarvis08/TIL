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