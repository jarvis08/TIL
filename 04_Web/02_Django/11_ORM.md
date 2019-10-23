# ORM

## ORM, Object-Relational Mapping

- ORM, Object-Relational Mapping

  DB의 행/테이블 등을 **객체**로 취급하여 Programming Language를 이용해 data 조작(SQL을 code로 사용)

  `Database <--SQL Statement-- | ORM | --Python Object--> Code`

  - Wekipedia

    - ORM은 객체 지향 프로그래밍 언어를 사용하여 호환되지 않는 유형의 시스템간에(Django - SQL)데이터를 변환하는 프로그래밍 기술

    - 프로그래밍 언어에서 사용할 수 있는 '가상 객체 데이터베이스'를 만들어 사용

- **사용 이점**

  Python의 Class로 DB를 조작 가능

  - SQL 문법을 몰라도 Query(데이터베이스에 정보를 요청) 조작 가능
  - 객체 지향적인 접근 가능 (인스턴스 / 클래스 변수 etc.)
  - 해당 객체의 재활용 가능

- DML(Data Manipulation Language)에서의 **CRUD**

  | SQL         | CRUD   |
  | ----------- | ------ |
  | SELECT      | READ   |
  | INSERT INTO | CREATE |
  | UPDATE      | UPDATE |
  | DELETE      | DELETE |

<br>

<br>

## DB 생성하기

- `settings.py` 에서 Default로 SQLite가 설정되어 있으며,

  프로젝트 생성 시 자동으로 `PROJECT_NAME/db.sqlite3`파일이 생성

  `db.sqlite3` 파일 하나로 DB 역할 수행

- `AppName/models.py`를 사용하여 데이터를 관리

  ```python
  # models.py
  from django.db import models
  
  class Article(models.Model):
      ## Scheme 정의
      # column을 정의하는 객체를 TextField() Class로 설정하여 속성임을 명시
      title = models.TextField()
      content = models.TextField()
      created_at = models.DateTimeField(auto_now_add=True)
  ```

<br>

### Migration

> Django에서 선언한 Model을 실제 Database에 반영하는 과정

1. `model.py` 내용물을 이용해 `migrate` 할 `AppName/migrations/내용물` 생성하기

   `migrations` : 상세 명세서와 같은 설계도, 청사진

   ```bash
   $ python manage.py makemigrations
   ```

   ```
   # 결과
   Migrations for 'articles':
     articles\migrations\0001_initial.py
       - Create model Article
   AppName/__init__.py
   AppName/0001_initial.py
   ```

2. Django에게 `migrate` 지시

   ```bash
   $ python manage.py migrate
   ```

3. `SQLite` Extention 혹은 https://inloop.github.io/sqlite-viewer/ 통해 `db.sqlite3`의 내용물 확인 가능

<br>

### Migration 갱신하기

- 갱신한 `models.py` 내용

  ```python
  # model.py
  from django.db import models
  
  class Article(models.Model):
      title = models.TextField()
      content = models.TextField()
      created_at = models.DateTimeField(auto_now_add=True)
  	# img_url 추가
      img_url = models.TextField()
  ```

  1. Column을 추가한 상태로 `makemigrations` 시도
  
       ```bash
       $ python manage.py makemigrations
       ```
       
2. 이미 있는 Database를 훼손할 수 있으므로 다음의 선택지들을 제시

        `1) Default 값을 지금 부여`
        
        `2) 취소 후 models.py에 default 값 설정하여 재시도`

      ```bash
      You are trying to add a non-nullable field 'img_url' to article without a default; we can't do that (the database needs something to populate existing rows).
      Please select a fix:
      1) Provide a one-off default now (will be set on all existing rows with a null value for this column)
      2) Quit, and let me add a default in models.py
      Select an option: #1
      ```

  3. Default Value 부여하기

       `''` 라는 default 부여

       ```bash
       Select an option: #1
       Please enter the default value now, as valid Python
       The datetime and django.utils.timezone modules are available, so you can do e.g. timezone.now
       Type 'exit' to exit this prompt
       >>> ''
       ```

       - 결과

         다음 파일 추가됨

         `AppName/migrations/0002_article_img_url.py`

         `AppName/migrations/` 내용을 통해 **어떻게 기능들이 추가되어 왔는지 History 확인 가능**

4. 수정 사항 적용하기

      ```bash
      $ python manage.py migrate
      ```

<br>

### SQLite 확인해보기

```bash
$ python manage.py sqlmigrate articles 0001
```

- 결과

    ```bash
    BEGIN;
    --
    -- Create model Article
    --
    CREATE TABLE "articles_article" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "title" text NOT NULL, "content" text NOT NULL, "created_at" datetime NOT NULL);
    COMMIT;
    ```

<br>

### DB 삭제 후 새로 데이터 생성하기

1. `PROJECT_NAME/db.sqlite3` 파일 삭제

2. Migrate

   ```bash
   $ python manage.py migrate
   ```

   ```bash
   $ python manage.py shell
   >>> from articles.models import Article
   >>> article = Article()
   >>> article.title = "첫 번째 글입니다."
   >>> article.content = "이건 내용입니다."
   >>> article.save()
   >>> Article.objects.all()
   <QuerySet [<Article: Article object (1)>]>
   ```

3. `__str__` 추가하기

   ```python
   from django.db import models
   
   class Article(models.Model):
       ## Scheme 정의
       # column을 정의하는 객체를 초기값으로 설정하여 속성임을 명시
       title = models.TextField()
       content = models.TextField()
       created_at = models.DateTimeField(auto_now_add=True)
       img_url = models.TextField()
   
       def __str__(self):
           return f'{self.id} | {self.title}'
   ```

   - `id` 값은 django가 알아서 생성하는 속성

     따라서 migrate를 해주지 않아도 된다

4. 확인하기

   ```bash
   >>> from articles.models import Article
   >>> Article.objects.all()
   <QuerySet [<Article: 1 | 첫 번째 글>]>
   # article 이라고 이전에 사용했던 변수명을 설정해도 무관
   # save()를 할 때마다 변수가 binding 되므로 변수명은 무엇이든 상관 없다.
   >>> article = Article(title='두 번째 글', content='냉무')
   >>> article.save()
   >>> Article.objects.all()
   <QuerySet [<Article: 1 | 첫 번째 글>, <Article: 2 | 두 번째 글>]>
   ```

<br>

<br>

## DB 갱신하기

1. `models.py`를 통해 DB data 불러오기

   `models.py` 전체를 불러오면 불필요한 data도 `import`

   따라서 `ClassName`을 지정하여 `import`

   ```python
   # views.py
   # from . import models
   from .models import Article
   ```

2. `views.py` 함수 수정

   `reversed(objects)`를 사용하여 최신 게시글부터 나열

   (실제로는 이렇게 사용하지 않는다)

   ```python
   def index(request):
       # context = {
       #     'blogs': blogs,
       # }
       articles = Article.objects.all()
       context = {
           'articles': reversed(articles),
       }
       return render(request, 'index.html', context)
   ```

3. HTML 파일 수정

   ```html
   {% for article in articles %}
     <p>제목: {{ article.title }}</p>
     <p>내용: {{ article.content }}</p>
     이미지: <img src="{{ article.img_url }}">
     <p>작성일자: {{ article.created_at }}</p>
   {% endfor %}
   ```

4. Web에서 새로 객체 생성할 수 있도록 수정하기

   ```python
   def create(request):
       # created_at = datetime.now()
       # datetime은 Article 객체 생성 시 자동 생성
       title = request.GET.get('title')
       content = request.GET.get('content')
       img_url = request.GET.get('img_url')
   
       # DB에 저장하기
       article = Article()
       article.title = title
       article.content = content
       article.img_url = img_url
       article.save()
   
       # blogs.append({'title': title, 'content': content, 'created_at': created_at})
       # 객체로 만들기
       # blogs.append(Article(title, content, created_at))
       
       context = {
           'title': title,
           'content': content,
           'img_url': img_url,
           'created_at': article.created_at,
       }
       
       return render(request, 'create.html', context)
   ```

<br>

### 객체를 DB에 저장하는 방법 네 가지

1. 인스턴스 생성 후 속성 별로 부여하기

   ```python
   def create(request):
       post = Post()
       post.title = request.GET.get('title')
       post.content = request.GET.get('content')
       post.img_url = request.GET.get('img_url')
       post.save()
   ```

2. 인스턴스 생성과 동시에 인수로 설정하기

   ```python
   def create(request):
       post = Post(
           title = request.GET.get('title')
           content = request.GET.get('content')
           img_url = request.GET.get('img_url')
       )
       post.save()
   ```

3. `save()`가 필요 없는 방법

   하지만 데이터 별 조건(불순 데이터 여부)을  만족하는지 확인하는 validation 과정이 대게 `save()`에 존재하므로, 좋은 방법이 아니다.

   ```python
   def create(request):
       Post.objects.create(
           title = request.GET.get('title')
           content = request.GET.get('content')
           img_url = request.GET.get('img_url')
       )
   ```

4. 세 번째 방법의 간편화 버전

   dictionary는 아니지만, keyword argument와 같은 방식이 가능

   단, DB의 Column 이름과 GET을 통해 가져온 data의 `name`이 동일해야 가능

   이 역시 validation 과정이 없어 좋은 방법은 아니다.

   ```python
   def create(request):
   	Post.objects.create(**request.GET)
   ```

<br>

<br>

## Lazy Loading

**DB를 `fetch` 단계 이전까지 접근하지 않으며, `fetch` 단계에서 몰아서 작업합니다**

사실상 `{% with 변수=코드 %}`를 사용하지 않아도 됩니다. 위에서는 `{% articles.objects.all %}`과 같이 데이터 베이스에서 작업하는 내용을 최소화 하기 위해 변수에 저장한다는 느낌으로 `{% with %}`를 사용했었습니다.

하지만 ORM은 Lazy Loading이라는 아주 효율적인 방법을 사용하여 Database를 다룹니다. 따라서 실제로 `{% with %}`는  코드 작성의 번거로움을 줄이는 용도로만 사용됩니다.

