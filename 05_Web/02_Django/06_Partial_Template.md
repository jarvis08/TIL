# Partial View, Rendering, Template

---

- Partial Temlplate

  **파일명** 앞에 **`_`**를 붙이는 것이 Convention

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