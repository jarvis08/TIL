# Meta Class

## 내림차순 정렬

내림차순으로 정렬하는 방법에는 세 가지가 있다.

1. HTML 파일에서 reversed
2. views.py에서 reversed
3. Meta Class 생성

<br>

### HTML 파일에서 정렬

```html
  {% for comment in comments reversed %}
    <p class="form-control pb-2">{{ comment.content }}</p>
  {% endfor %}
```

<br>

### Meta Class

`views.py`에는 최소한의 data 전송 코드만을 작성하는게 좋다. 따라서 `views.py`에서 `reversed`를 사용하는 것은 옳지 않으며, `models.py` 에서 게시글 생성 시간을 기준으로 `Meta`  클래스([Django Meta Options](https://docs.djangoproject.com/en/2.2/ref/models/options/))를 정의하여 `ordering`옵션 부여.

```python
# models.py
class Post(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-pk']
        

class Comment(models.Model):
    content = models.CharField(max_length=300)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    post = models.ForeignKey(Post, on_delete=models.CASCADE)

    class Meta:
        ordering = ['-pk']
```

클래스 안에 클래스를 집어넣는 하위 개념은 아니다.

이 작업은 DB를 변경하는 것이 아니라, ORM 개념이므로 `migrate`를 다시 할 필요 없다.