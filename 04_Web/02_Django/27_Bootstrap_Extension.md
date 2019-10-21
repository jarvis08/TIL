# Bootstrap Extension

**CDN(Content Delivery Network)을 사용하지 않는**, 서버에 설치하는 방법입니다.

<br>

<br>

### Install

`pip install django-bootstrap4`

```python
INSTALLED_APPS = [
    'articles',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'bootstrap4',
]
```

<br>

<br>

## HTML에서 사용하기

### base.html 설정

bootstrap을 사용할 것이라고 설정하는 코드입니다.

```html
{% load bootstrap4 %}
{% bootstrap_css %}
{% bootstrap_javascript jquery='full' %}
```

설정 예시입니다.

```html
<!-- base.html -->
{% load bootstrap4 %}
{% bootstrap_css %}
{% bootstrap_javascript jquery='full' %}

<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="ie=edge">
  <title>Document</title>
</head>
<body>
  {% block body %}
  {% endblock %}
</body>
</html>
```

<br>

### 직접 사용되는 HTML

bootstrap을 사용할 페이지에서 bootstrap을 불러온 후(`{% load bootstrap4 %}`), `{% block body %}`의 `block` 과 같이, `{% bootstrap_form form%}`을 사용하여 form을 생성합니다. bootstrap을 사용하지 않을 경우 `{{ form }}`으로 사용했었습니다.

```html
{% load bootstrap4 %}
{% bootstrap_form form %}
```

사용 예시입니다.

```html
<!-- create.html -->
{% extends 'base.html' %}

{% load bootstrap4 %}

{% block body %}
<h1>NEW</h1>
<form action="{% url 'articles:create' %}" method="POST">
  {% csrf_token %}
    
  {% bootstrap_form form %}

  <br>
  <button type="submit">제출</button>
</form>
<a href="{% url 'articles:index' %}">뒤로가기</a>
{% endblock %}
```