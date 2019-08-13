# Add Function to an Existing Apps

---

## 이미 존재하는 App에 기능을 추가하는 과정

1. `urls.py`에 `url`과  `view 함수 위치` 추가
2. `views.py`에 함수 작성
3. 함수에 대한 html파일 제작

---

- 숫자 변수 받기

  - `urls.py`

    ```python
    from django.contrib import admin
    from django.urls import path
    from AppName import views
    
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

---

## input 변수 전달하기

### 1. GET

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
  <p>{{ path_1 }}</p>
  <p>{{ path_2 }}</p>
  <p>{{ scheme }}</p>
  <p>{{ method }}</p>
  <p>{{ host }}</p>   
  ```
  
  ```
  ## result
  조동빈님과 박현지님의 궁합은 100%입니다.
  /match/
  /match/
  http
  GET
  localhost:8000
  ```

### 2. POST

- `index.html`의 **`POST`** 방식의 `<form>`

  `POST`의 경우 `CSRF Token`으로 넣어버리는 것과, 무시하는 방법이 존재

  Token으로 넣을 시 다음의 코드를 추가, `{% csrf_token %}`

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
  조동빈님과 박현지님의 궁합은 100%입니다.
  POST
  ```

