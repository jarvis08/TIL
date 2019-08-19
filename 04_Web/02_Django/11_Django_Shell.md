# Django Shell

## Django Shell 사용하여 DB 조작하기

### 객체 생성, 저장, 확인

```bash
$ python manage.py shell
>>> from articles.models import Article
# Article의 객체 다 꺼내보기
>>> Article.objects.all()

# 객체 생성 방법 1
>>> article = Article()
>>> article.title = "첫 번째 글입니다."
>>> article.content = "이건 내용입니다."

# 아직 빈 리스트이며, 저장이 필요
>>> Article.objects.all()
Traceback (most recent call last):
  File "<console>", line 1, in <module>
AttributeError: type object 'Article' has no attribute 'object'

# 저장 후 확인
>>> article.save()
>>> Article.objects.all()
<QuerySet [<Article: Article object (1)>]>
# 저장된 객체는 리스트처럼 다룰 수 있다

# 저장 된 첫 번째 객체 조회하기
>>> first_article = Article.objects.first()
>>> first_article.title
>>> first_article.image_url
>>> first_article.created_at

# 객체 생성 방법 2
>>> article_two = Article(title='2nd', content='initialize')
>>> article_two.save()
>>> Article.objects.all()
<QuerySet [<Article: Article object (1)>, <Article: Article object (2)>]>
```