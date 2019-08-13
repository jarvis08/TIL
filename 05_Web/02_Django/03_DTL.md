# Django Template Language

---

## Django's Conventions

- url은 항상 **`/`**로 종료

- 마지막 요소일 지라도 `,`를 붙여주기

- `views.py` 함수의 parameter로 항상 **`request`**를 기입

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

---

## for 문

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

---

## if 문

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

---

## helper $ filter

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
    

---

## filter

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


---

## datetime

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

---

## extends

- Template 상속하여 편의성 급증

  e.g., Bootstrap을 Template마다 적용시키기

---

## block