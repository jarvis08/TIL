# DTL, Django Template Language

## Template 이동

### 새 페이지 Load

- `render(request, url, 변수Dict)` 형태로 사용하며,

  `render()` 함수는 `HttpResponse`를 생성하여 반환

  e.g., `render(request, 'home.html', context)`

  - 변수들을 묶어서 `context`라는 **dictionary** 변수 안에 포함시켜 한번에 `render()`의 인수로 기입

  - `context`라는 단어는 무엇을 사용하든 무관하지만,

    **context**를 넘긴다는 의미 하에 관례적으로 사용되며, `ctx`로 줄여서 표현

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

<br>

### 이전의 특정 페이지 Load

- `redirect('별명')`

  존재하는 다른 페이지로 돌아가도록 설정하기
  
  1. `PROJECT_NAME/project_name/urls.py`에 Url의 별명을 설정
  
     ```python
     # PROJECT_NAME/project_name/urls.py
     urlpatterns = [
         path('admin/', admin.site.urls),
         # path('', views.index),
         path('', views.index, name='index'),
     ]
     ```
     
  2. `views.py`에 `render()` 대신 `redirect()` 사용하기
  
     `urls.py`에서 설정한 `name`을 사용
     
     ```python
     # views.py
     from django.shortcuts import render, redirect
     
     def create(request):
         return redirect('index')
     ```

<br>

<br>

## Template 내부 제어

### for 문

```html
{% for s in lotto %}
	<p>{{ s }}</p>
{% endfor %}
```

- 2중 for문도 가능

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

<br>

### if 문

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

<br>

### helper

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
    

<br>

### filter

- `{{ html변수|filter함수 }}`
  - `length`

    길이 표시

  - `truncatechars:value`

    글자수를 `value`만큼 제한하여, 나머지는 `...` 처리

      ```html
    {% for movie in empty_dataset %}
      <p>{{ movie|length }} {{ movie|truncatechars:5 }}</p>
    {% endfor %}
      ```

<br>

### extends & include

- `extends`

  `block`과 함께 사용되며, 현재 template의 내용을 `base.html`의 `block` 부분에 삽입한 형태로 사용

  ```html
  {% extends 'base.html' %}
  {% block block_name %}
  내용
  {% endblock %}
  ```

- `inclunde`

  `_navbar.html`의 내용을 `{% inlude %}`의 위치에 삽입
  
  ```html
  {% include '_navbar.html' %}
  ```

<br>

### block

- 다른 template의 내용을 끼워 넣는 것에 사용

  ```html
  {% extends 'base.html' %}
  {% block block_name %}
  내용
  {% endblock %}
  ```

<br>

<br>

## Date & Time

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
  <!-- 출력 -->
  20
  ```

  But `가능 != 해도된다`

<br>

### 1993-01-05 형태

```html
<input type="date" name="created_at" value="{{ posts.created_at|date:"Y-m-d" }}">
```

<br>

### 더욱 편리하게 사용하기

```html
{% extends 'base.html' %}
{% block body %}
  <h1>DETAIL</h1>
  <p>Title: {{ article.title }}</p>
  <p>Content: {{ article.content }}</p>
  <p>Created at: {{ article.created_at | date:"Y년 m월 d일" }}</p>
  <p>Updated at: {{ article.updated_at | date:"SHORT_DATE_FORMAT" }}</p>
  <a href="{% url 'articles:index' %}">글 목록</a>
{% endblock %}
```

![Date_Format](assets/Date_Format.jpg)