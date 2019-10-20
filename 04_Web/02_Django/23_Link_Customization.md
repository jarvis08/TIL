# Link Customization

### 링크(url) 내 마음대로 사용하기

```html
<!-- index.html 수정 이전 -->
<a href="{% url 'articles:detail' article.pk %}">상세보기</a>
```

위 방식은 너무 보기에 지저분하다. 따라서 models.py에 메써드를 추가하여 단축시킬 수 있다. 실질적으로는 이미 가지고 있는 메써드(`urls`의 `reverse()`)를 **오버라이드**한다.

```python
from django.db import models
from django.urls import reverse

# Create your models here.
class Article(models.Model):
    title = models.CharField(max_length=50)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
	
    # 편리하게 title을 표시하기 위함
    def __str__(self):
        return self.title
    
    # 게시글을 역순으로 저장(최신 글이 위로 오도록)
    class Meta:
        ordering = ('-pk',)
    
    def get_absolute_url(self):
        # reverse('어느 뷰 함수로 가는지', '인자')
        return reverse('articles:detail', kwargs={'article_pk': self.pk})
```

```html
<!-- index.html 수정 이후 -->
<a href="{{ article.get_absolute_url }}">상세보기</a>
```

<br>

이를 이용하여 `redirect` 메서드도 변화시킬 수 있다. 내부적으로 `resolve_url()`을 사용하는데, `resolve_url()`은 `reserve()`를 사용한다. 따라서 models.py에 `reverse()`를 사용하는 `get_absolute_url()`을 설정해 놓았으므로 `redirect()`에 `article` 객체를 넣는 것 만으로도 `redirect()`가 가능하다.

```python
# views.py, redirect 수정 이전
def create(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        content = request.POST.get('content')
        article = Article.objects.create(title=title, content=content)
        ##############################################
        return redirect('articles:detail', article.pk)
    	##############################################
    else:
        return render(request, 'articles/create.html')
```

```python
# views.py, redirect 수정 이후
def create(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        content = request.POST.get('content')
        article = Article.objects.create(title=title, content=content)
        ########################
        return redirect(article)
    	########################
    else:
        return render(request, 'articles/create.html')
```

