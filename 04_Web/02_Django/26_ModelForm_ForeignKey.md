# ModelForm Application

## Comments

### Urls

```python
# urls.py

urlpatterns = [
    path('', views.index, name='index'),
    path('<int:article_pk>/', views.detail, name='detail'),
    path('<int:article_pk>/update/', views.update, name='update'),
    path('<int:article_pk>/delete/', views.delete, name='delete'),
    path('create/', views.create, name='create'),
    path('<int:article_pk>/comments/', views.comments, name='comments'),
    path('<int:article_pk>/comments_delete/<int:c_id>/', views.comments_delete, name='comments_delete'),
]
```

<br>

### Model

```python
class Comment(models.Model):
    content = models.TextField()
    created_at= models.DateTimeField(auto_now_add=True)
    updated_at= models.DateTimeField(auto_now=True)
    article = models.ForeignKey(Article, on_delete=models.CASCADE)

    class Meta:
        ordering = ('-pk',)
```

<br>

### ModelForm

```python
# forms.py
from .models import Comment

class CommentModelForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ('content',)
    
    content = forms.CharField(
        label='댓글',
        widget=forms.Textarea(
            attrs={
                'class': 'form-control my-content',
                'placeholder': '댓글을 입력해주세요.',
                'rows': 5,
            }
        )
    )
```

<br>

### Views

```python
# views.py
from django.shortcuts import render, redirect, get_object_or_404
from .models import Article, Comment
from .forms import ArticleForm, ArticleModelForm, CommentModelForm
from django.http import Http404
from django.views.decorators.http import require_POST

def detail(request, article_pk):
    # article = get_object_or_404(Article, pk=article_pk)
    # 직접 만들어 보기
    try:
        article = Article.objects.get(pk=article_pk)
    # except:
    ## 에러 지정하기
    except Article.DoesNotExist:
        raise Http404('해당하는 ID의 글이 존재하지 않습니다.')
    
    form = CommentModelForm()
    context = {
	    	'article': article,
    		'form': form,
      	'comments': article.comment_set.all(),
    }
    return render(request, 'articles/detail.html', context)


def comments(request, article_pk):
  	article = Article.objects.get(pk=article_pk)
    if request.method == 'POST':
        form = CommentModelForm(request.POST)
        
        if form.is_valid():
          	# commit: Database Commit
            # Foreign Key가 아직 등록되지 않았으므로, DB에 바로 넣으려 하면 에러 발생
            comment = form.save(commit=False)
            comment.article = article
            # comment.article_id = article_id
            comment.save()
    return redirect(article)


@require_POST
def comments_delete(request, article_pk, c_id):
    comment = get_object_or_404(Comment, pk=c_id)
    if request.method == 'POST':
        comment.delete()
    return redirect('articles:detail', article_pk)
```

<br>

### HTML

```html
{% load bootstrap4 %}
{% block body %}
  <div class="container">
    <h1 class="text-center">DETAIL</h1>
    <a href="{% url 'articles:index' %}">목록</a>
    
    <br>
    <label>Title</label>
    <p class="form-control">{{ article.title }}</p>
    <label>Content</label>
    <p class="form-control">{{ article.content }}</p>
    <label>Created at</label>
    <p class="form-control">{{ article.created_at | date:"Y년 m월 d일" }}</p>
    <label>Updated at</label>
    <p class="form-control">{{ article.updated_at | date:"SHORT_DATE_FORMAT" }}</p>

    <a href="{% url 'articles:update' article.pk %}">수정</a>
    <form class="d-inline" action="{% url 'articles:delete' article.pk %}" method="POST">
      {% csrf_token %}
      <button type="submit">삭제</button>
    </form>

    <br>
    <br>
    <br>
    <h2>COMMENTS</h2>

    <table class="table">
      <thead>
        <tr>
          <th scope="col">Content</th>
          <th scope="col">actions</th>
        </tr>
      </thead>
      <tbody>
        {% for comment in comments %}
          <tr>
            <td>{{ comment.content }}</td>
            <td><form action="{% url 'articles:comments_delete' article.pk comment.pk %}" method="POST">
                {% csrf_token %}
                <button type="submit">삭제</button></form></td>  
          </tr>
        {% endfor %}
      </tbody>
    </table>

    <form action="{% url 'articles:comments' article.pk %}" method="POST">
      {% csrf_token %}
      {% bootstrap_form form %}
      <button type="submit">댓글달기</button>
    </form>
  </div>
{% endblock %}
```

![comments_snapshot](./assets/comments_snapshot.JPG)