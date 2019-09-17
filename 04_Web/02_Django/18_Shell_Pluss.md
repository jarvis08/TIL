# Shell Plus

Django의 객체들을 shell에서 관리할 수 있는 extension

<br>

### 설치

`$ pip install django-extensions`

```python
# settings.py에 extensions 사용 지시
INSTALLED_APPS = [
    'django_extensions',
    'posts',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]
```

<br>

### 사용하기

`python manage.py shell_plus`

```shell
$ python manage.py shell_plus
# Shell Plus Model Imports
from django.contrib.admin.models import LogEntry
from django.contrib.auth.models import Group, Permission, User
from django.contrib.contenttypes.models import ContentType
from django.contrib.sessions.models import Session
from posts.models import Comment, Post
# Shell Plus Django Imports
from django.core.cache import cache
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import transaction
from django.db.models import Avg, Case, Count, F, Max, Min, Prefetch, Q, Sum, When, Exists, OuterRef, Subquery
from django.utils import timezone
from django.urls import reverse
Python 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 19:29:22) [MSC v.1916 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license" for more information.
(InteractiveConsole)
>>> 
```

```shell
# Post 객체 확인
>>> Post.objects.all()
<QuerySet [<Post: Post object (2)>, <Post: Post object (3)>, <Post: Post object (4)>, <Post: Post object (5)>,
<Post: Post object (6)>]>
```

```shell
# 가장 최근에 생성된 객체 확인
>>> Post.objects.last().title
'포도'
```

```shell
# 새로운 객체 생성
>>> Post.objects.create(title='하이', content='하이하이')
<Post: Post object (7)>
>>> Post.objects.last().title
'하이'
```

```shell
# 최근 객체에 Comment 달아보기(틀린 방법)
>>> Comment.objects.create(content='첫 댓글', post=7)
ValueError: Cannot assign "7": "Comment.post" must be a "Post" instance.
# Comment의 post 변수에는 Post의 객체를 지정해야 하므로 오류
```

```shell
# Post 객체를 외래키로 지정하여 Comment 객체 생성하기
>>> post_7 = Post.objects.get(pk=7)
>>> Comment.objects.create(content='첫 댓글', post=post_7)
<Comment: Comment object (1)>
```

```shell
# Comment 객체 다루기
>>> comm_1 = Comment.objects.first()
>>> comm_1.content
'첫 댓글'
```

```shell
# Comment 객체를 통해 Post 객체 수정하기
>>> comm_1.post.content
'하이하이'
>>> comm_1.post.content = '수정된 하이하이'
>>> comm_1.post.content
'수정된 하이하이'
>>> comm_1.post.save()
```

```shell
# Meta Programming : 자동으로 Class 이름인 Comment를 comment로 변경하여 함수로 사용
# 객체명.소문자클래스명_set.all()
>>> post_7.comment_set.all()
<QuerySet [<Comment: Comment object (1)>]>

# 뫼뷔우스의 띠처럼 계속 돌기
>>> post_7.comment_set.last().content
'첫 댓글'
>>> post_7.comment_set.last().post
<Post: Post object (7)>
>>> post_7.comment_set.last().post.comment_set.last().content
'첫 댓글'
>>> post_7.comment_set.last().post.comment_set.last().post.comment_set.last().post.comment_set.last().content
'첫 댓글'
```

