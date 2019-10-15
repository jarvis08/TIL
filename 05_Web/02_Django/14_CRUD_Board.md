# CRUD의 활용, Bulletin Board 만들기

## CRUD

> Create
>
> Read
>
> Update
>
> Delete

<br>

### 게시판 생성

```python
# views.py
def index(request):
    posts = Post.objects.all()
    context = {
        'posts': reversed(posts),
    }
    return render(request, 'posts/index.html', context)
```

```html
<!-- index.html -->
{% extends 'base.html' %}
{% block body %}
<div class="container">
  <h1>SSAFY 게시판</h1>
  <table class="table table-striped">
    <thead>
      <tr>
        <th scope="col">ID</th>
        <th scope="col">제목</th>
        <th scope="col">생성일</th>
        <th scope="col">수정일</th>
      </tr>
    </thead>
    <tbody>
      {% for post in posts %}
      <tr>
        <th scope="row">{{ post.id }}</th>
        <td>{{ post.title }}</td>
        <td>{{ post.created_at }}</td>
        <td>{{ post.updated_at }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>
{% endblock %}
```

<br>

### Create, 작성하기

```python
# views.py
def new(request):
    return render(request, 'posts/new.html')

def create(request):
    post = Post()
    post.title = request.GET.get('title')
    post.content = request.GET.get('content')
    post.img_url = request.GET.get('img_url')
    post.save()
    return redirect('home')
```

```html
<!-- new.html -->
{% extends 'base.html' %}
{% block body %}
<div class="container">
  <h1>새 글 쓰기</h1>

  <form action="{% url 'posts:create' %}" method="GET">
    <div class="form-group">
      <label for="title">제목</label>
      <input type="text" class="form-control" id="title" name="title" placeholder="제목을 입력해 주세요.">
      <label for="content">내용</label>
      <textarea name="content" class="form-control" id="content" rows="10" placeholder="내용을 입력해 주세요."></textarea>
      <label for="img_url">이미지 URL</label>
      <input type="text" class="form-control" id="img_url" name="img_url" placeholder="이미지 URL을 입력해 주세요.">
    </div>
    <button type="submit" class="btn btn-primary">글 쓰기</button>
  </form>
</div>
{% endblock %}
```

<br>

### Read, 세부 내용 보기

- `title` 에 link 부여하여 게시글 보기(detail)

  id를 주소로 사용하는 페이지로 전송하면, 해당 페이지를 보여주도록 설정

```html
<!-- index.html -->
{% extends 'base.html' %}
{% block body %}
<div class="container">
  <h1>SSAFY 서울 1반 게시판</h1>
  <table class="table table-striped">
    <thead>
      <tr>
        <th scope="col">ID</th>
        <th scope="col">제목</th>
        <th scope="col">생성일</th>
        <th scope="col">수정일</th>
      </tr>
    </thead>
    <tbody>
      {% for post in posts %}
      <tr>
        <th scope="row">{{ post.id }}</th>
        <td><a href="/posts/{{ post.id }}/">{{ post.title }}</a></td>
        <td>{{ post.created_at }}</td>
        <td>{{ post.updated_at }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>
{% endblock %}
```

```python
# AppName/urls.py
urlpatterns = [
    path('new/', views.new, name='new'),
    path('create/', views.create, name='create'),
    path('<int:pk>/', views.detail, name='detail'),
]
```

```python
# views.py
def detail(request, pk):
    post = Post.objects.get(pk=pk)
    context = {
        'post': post,
    }
    return render(request, 'posts/detail.html', context)
```

```html
<!-- detail.html -->
<div class="container">
  <a href="/posts/{{ post.pk }}/edit/" class="btn btn-success">수정</a>
  <a href="/posts/{{ post.pk }}/delete/" class="btn btn-danger">삭제</a>
  <h1>{{ post.title }}</h1>
  <p>{{ post.content }}</p>
  <img src="{{ post.img_url }}" alt="">
  <p>{{ post.created_at }}</p>
  <p>{{ post.updated_at }}</p>
</div>
```

```html
<!-- 별명 이용하여 link 보내는법 -->
<a href="{% url 'posts:edit' post.pk %}" class="btn btn-success">수정</a>
<a href="{% url 'posts:elete' post.id %}" class="btn btn-danger">삭제</a>
```

<br>

### Delete, 삭제하기

```python
# urls.py
urlpatterns = [
    path('new/', views.new, name='new'),
    path('create/', views.create, name='create'),
    path('<int:pk>/', views.detail, name='detail'),
    path('<int:pk>/delete/', views.delete, name='delete'),
]
```

```python
# views.py
def delete(request, pk):
    # 지우는 방법 두 가지
    # 1. 찾아서 삭제
    post = Post.objects.get(pk=pk)
    post.delete()

    # # 2. 바로 삭제
    # Post.objects.get(pk=pk)
    return redirect('home')
```

<br>

### Update, 수정하기

1. 수정을 하기 위한 페이지

   새 글 작성과 유사하지만, 기존의 내용이 모두 기록되어 있음(`value="내용"` 부여)

   ```python
   # urls.py
   urlpatterns = [
       path('new/', views.new, name='new'),
       path('create/', views.create, name='create'),
       path('<int:pk>/', views.detail, name='detail'),
       path('<int:pk>/delete/', views.delete, name='delete'),
       path('<int:pk>/edit/', views.edit, name='edit'),
   ]
   ```

   ```python
   # views.py
   def edit(request, pk):
       # pk라는 id를 가진 글을 편집
       post = Post.objects.get(pk=pk)
       context = {
           'post': post,
       }
       return render(request, 'posts/edit.html', context)
   ```

   ```html
   <!-- edit.html -->
   {% extends 'base.html' %}
   {% block body %}
   <div class="container">
     <h1>수정하기</h1>
     <!-- <form action="posts/{{ posts.pk }}/update/" method="GET"> -->
     <form action="{% url 'posts:update' post.id %}" method="GET">
       <div class="form-group">
         <label for="title">제목</label>
         <input type="text" class="form-control" id="title" name="title" value="{{ post.title }}">
         <label for="content">내용</label>
         <textarea name="content" class="form-control" id="content" rows="10">{{ post.content }}</textarea>
         <label for="img_url">이미지 URL</label>
         <input type="text" class="form-control" id="img_url" name="img_url" value="{{ post.img_url }}">
       </div>
       <button type="submit" class="btn btn-primary">수정하기</button>
     </form>
   </div>
   {% endblock %}
   ```

   - 1993-02-05 형태로 이전의 날짜 데이터 불러오기

     ```html
     작성날짜 : <input type="date" name="created_at" value="{{ posts.created_at|date:'Y-m-d' }}">
     ```

2. 수정 사항을 DB에 반영하는 위한 `update`

   반영한 이후에는 다시 상세 정보 페이지(Read)로 이동
   
   ```python
   # urls.py
   urlpatterns = [
       path('new/', views.new, name='new'),
       path('create/', views.create, name='create'),
       path('<int:pk>/', views.detail, name='detail'),
       path('<int:pk>/delete/', views.delete, name='delete'),
       path('<int:pk>/edit/', views.edit, name='edit'),
       path('<int:pk>/update/', views.update, name='update'),
]
   ```
   
   ```python
   # views.py
   def update(request, pk):
       # 1. pk라는 id를 가진 글 찾기
       # 2. /edit/으로부터 날아온 데이터를 적용하여 변경
       post = Post.objects.get(pk=pk)
       post.title = request.GET.get('title')
       post.content = request.GET.get('content')
       post.img_url = request.GET.get('img_url')
       post.save()
       # return redirect(f'/posts/{pk}/')
       return redirect('posts:detail', pk)
   ```
   
   