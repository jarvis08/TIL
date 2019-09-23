# Database Relation

## 관계의 종류

1. No Relation
2. `1:1`
3. `1:N`
4. `N:N`

<br>

### 관계 없음

아무런 관계가 존재하지 않는 관계

<br>

### 1:1 관계

DTL Model에서 1:1 관계는 다음과 같이 표현한다.

`models.OneToOne(모델명, on_delete=)`

`on_delete` 매개변수는 관계를 맺는 대상이 삭제될 경우 어떻게 처리할 것인지를 설정한다.

`on_delete=models.CASCADE` : 참조하는 대상까지 삭제

<br>

### 1:N 관계

`1:N`, `One to Many`, `N:1`, `Many to One` 모두 같은 관계를 뜻하며, 다음과 같이 말할 수 있다.

- `1` has `N`
- `N` belogs to `1`
- `N` 쪽에서 `1`에 대한 Foreign Key(외래키)를 가지고 조회

<br>

### N:N 관계

`Many to Many`은 `One to Many`를 활용하여 구현한다.

<br><br>

## Foreign Key

`외래키명 = models.ForeignKey(모델명, on_delete=삭제시변경방법)`

i.g., 하나의 게시글에 대해 여러 댓글이 존재하며, 댓글들은 게시글의 기본키(PK, Primary Key)를 외래키(FK, Foreign Key)로 보유

```python
class Post(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class Comment(models.Model):
    content = models.CharField(max_length=300)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    # models.CASCADE : Post 삭제시 참조하는 Comment 모두 삭제
    post = models.ForeignKey(Post, on_delete=models.CASCADE)
```

```python
# views.py
def create_comment(request, pk):
    Comment.objects.create(
        content=request.POST.get('content'),
        post=Post.objects.get(pk=pk)
    )
    return redirect('posts:detail', pk)
```

<br>

### 확인해 보기

`$ python manage.py sqlmigrate posts 0002`

`0002`는 migrate 이후 migrations에 저장된 번호

```shell
$ python manage.py sqlmigrate posts 0002
BEGIN;
--
-- Create model Comment
--
CREATE TABLE "posts_comment" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "content" varchar(300) NOT NULL, "created_at" datetime NOT NULL, "updated_at" datetime NOT NULL, "post_id" integer NOT NULL REFERENCES "posts_post" ("id") DEFERRABLE INITIALLY DEFERRED);
CREATE INDEX "posts_comment_post_id_e81436d7" ON "posts_comment" ("post_id");
COMMIT;
```

migrate 할 때에는 외래키를 `post`라는 변수에 저장했지만,

migrate 이후 위의 shell 실행 결과에는 `post_id`에 저장된 것을 확인 가능