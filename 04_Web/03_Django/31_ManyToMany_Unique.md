# M:N with Unique Option

**Hash Tag**는 하나의 태그에 여러 게시글들이 할당되어야 합니다. 같은 값이 다른 객체를 가지면 안됩니다. 따라서 `unique=True` 값을 부여하여 동일 값으로 생성하려 했을 때 에러를 발생시킵니다. 일반적인 사용처럼 에러발생이 아닌, **화면에 동일 해시 태그를 보여주도록 하려면 Javascript를 사용**해야 합니다.

```python
# articles/models.py
from django.db import models
from django.conf import settings


class Hashtag(models.Model):
    content = models.TextField(unique=True)

    def __str__(self):
        return self.content
    
    
class Article(models.Model):
    title = models.CharField(max_length=50)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    like_users = models.ManyToManyField(settings.AUTH_USER_MODEL, related_name='like_articles', blank=True)
    hashtags = models.ManyToManyField(Hashtag, blank=True)

    def __str__(self):
        return self.title
    
    class Meta:
        ordering = ('-pk',)
    
    def get_absolute_url(self):
        # reverse('어느 뷰 함수로 가는지', '인자')
        return reverse('articles:detail', kwargs={'article_pk': self.pk})
```

```python
# articles/views.py
@login_required
def create(request):
    if request.method == 'POST':
        form = ArticleModelForm(request.POST)

        if form.is_valid():
            article = form.save(commit=False)
            article.user = request.user
            article.save()

            # hashtag
            for word in article.content.split():
                if word.startswith('#'):
                    # 기존 목록에 있는지 찾아본 후(가져오던가), 없으면 생성
                    # get_or_create()의 반환값 형태 :: (해당객체, True or False)
                    hashtag, created = Hashtag.objects.get_or_create(content=word)
                    article.hashtags.add(hashtag)
            return redirect(article)
        else:
            return redirect('articles:create')
    else:
        form = ArticleModelForm()
        context = {
            'form': form,
        }
        return render(request, 'articles/create.html', context)
```

`.startswith('string')`을 통해 #으로 시작하는 단어를 골라내며, `.get_or_create()`를 통해 있을 경우 가져오며, 없을 경우 `Hashtag` 객체를 생성합니다. `.get_or_creat()` 메서드는 `(객체, True/False)` 형태의 튜플을 반환합니다.

```html
<!-- detail.html -->
<h2>Hash Tags</h2>
<p><strong>
  {% for hashtag in article.hashtags.all  %}
    {{ hashtag }}
  {% endfor %}
</strong></p>
```

<br>

### Hash Tag로 글 모아보기

```python
# articles/urls.py
urlpatterns = [
    path('tags/', views.tags, name='tags'),
    path('hashtag/<int:hashtag_pk>/', views.hashtag, name='hashtag'),
]
```

```python
# articles/views.py
def tags(request):
    # 모든 해쉬태그 목록 보여주기
    tags = Hashtag.objects.all()
    context = {
        'tags': tags,
    }
    return render(request, 'articles/tags.html', context)


def hashtag(request, hashtag_pk):
    # 해당 해쉬태그가 들어간 해당하는 게시글
    hashtag = get_object_or_404(Hashtag, pk=hashtag_pk)
    articles = hashtag.article_set.all()
    context = {
        'hashtag': hashtag,
        'articles': articles,
    }
    return render(request, 'articles/hashtag.html', context)
```

```html
<!-- tags.html -->
{% extends 'base.html' %}

{% block body %}
<h1>Tags</h1>
{% for tag in tags %}
<a href="{% url 'articles:hashtag' tag.pk %}">{{ tag }}</a>
{% endfor %}
{% endblock %}
```

```html
<!-- hashtag.html -->
{% extends 'base.html' %}

{% block body %}
<h1>{{ hashtag.content }}</h1>
<ul>
  <hr>
  {% for article in articles %}
    <li><a href="{% url 'articles:detail' article.pk %}">{{ article.title }}</a></li>
    {{ article.content }} (Likes: {{ article.like_users.count }} | Comments: {{ article.comment_set.count }})
  {% endfor %}
</ul>
{% endblock %}
```

