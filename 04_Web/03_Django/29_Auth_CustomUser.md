# Customization of User Model

Instagram과 같은 SNS의 Follow 기능을 구현하려면 default User Model의 Customizing이 불가피합니다. `ManyToManyField`를 User Table에 정의해 주어야 하며, `AbstractBaseUser`를 커스터마이징 하기에는 너무 코드 작성량이 늘어나므로, 일단은 연습용으로 `AbstactUser`를 커스터마이징 해 보겠습니다.

```python
# accounts/models.py
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings

class User(AbstractUser):
    followers = models.ManyToManyField(settings.AUTH_USER_MODEL, related_name='followings', blank=True)
```

github에 공유된 django 코드를 참조해 봤을 때, 기본적으로 사용하는 사용자에 대한 모델은 `AUTH_USER_MODEL = 'auth.User'`로 default 값이 설정되어 있습니다. 이 default 모델은 django에서 제공하는 사용자 모델입니다. 따라서 우리가 커스터마이징한 모델을 사용도록 settings.py를 설정해주어야 합니다.

```python
# settings.py
AUTH_USER_MODEL = 'accounts.User'
```

User 모델을 새로 작성했기 때문에 현재까지 사용했던 **Database는 모두 reset**해주어야 합니다.

<br>

### Creation Form Customizing

하지만 이대로 사용하려 한다면, signup view에서 다음과 같은 에러가 발생합니다.

```
Manager isn't available; 'auth.User' has been swapped for 'accounts.User'
```

커스터마이징한 User 모델에 대해 Form이 존재하지 않기 때문에 발생합니다. `UserCreationForm`의 default model은 `User`로 지정되어 있으며, 이는 우리가 커스터마이징 하기 이전의 django model입니다.

따라서 다음과 같이 accounts/forms.py에 새로 클래스를 생성해 줍니다.

```python
# accounts/forms.py
from django.contrib.auth.forms import UserChangeForm, UserCreationForm


class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = get_user_model()
        fields = UserCreationForm.Meta.fields + ('email',)
```

이후 views.py에서 앞으로는 사용되지 않을 `UserCreationForm`을 삭제해 주며, Customizing된 form을 불러와 줍니다. 주석처리 된 곳은 수정 이전의 코드입니다.

```python
# accounts/views.py
# from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, UserChangeForm, PasswordChangeForm
# from .forms import CustomUserChangeForm
from django.contrib.auth.forms import AuthenticationForm, UserChangeForm, PasswordChangeForm
from .forms import CustomUserChangeForm, CustomUserCreationForm
```

아래는 `signup` view 함수의 내용 또한 수정해 줍니다.

```python
def signup(request):
    # 만약 로그인 되어있다면, index로 redirect
    if request.user.is_authenticated:
        return redirect('articles:index')
    if request.method == 'POST':
        # form = UserCreationForm(request.POST)
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('articles:index')
    else:
        form = UserCreationForm()
    context = {
        'form': form
    }
    return render(request, 'accounts/auth_form.html', context)
```

<br>

### 세부 구현

```python
# accounts/views.py
def follow(request, person_pk):
    person = get_object_or_404(User, pk=person_pk)
    user = request.user
    # user: 현재 로그인된 유저, 대상을 팔로우하고자 하는 유저
    if person.followers.filter(pk=user.pk).exists():
        person.followers.remove(user)
    else:
        person.followers.add(user)
    return redirect('profile', person.username)
```

```html
<!-- profile.html -->
{% with followers=person.followers.all followings=person.followings.all %}
  {% if user in followers %}
    <a href="{% url 'accounts:follow' person.pk %}" class="btn btn-outline-primary">Unfollow</a>
  {% else %}
    <a href="{% url 'accounts:follow' person.pk %}" class="btn btn-primary">Follow</a>
  {% endif %}
  <p>Follower: {{ followers|length }}</p>
  <p>>> List of followers -
    {% for follower in followers %}
      {{ follower }}
    {% endfor %}
  </p>
  <p>Following: {{ followings|length }}</p>
  <p>>> List of followings -
    {% for following in followings %}
      {{ following }}
    {% endfor %}
  </p>
{% endwith %}
```

<br>

<br>

## Following 하는 사람들의 글만 Feed(index)에서 보기

`articles = Article.objects.all()`을 통해 모든 게시글을 불러오는 것 과 달리, 작성자가 following 리스트에 있는 게시글들만을 불러옵니다.

```python
# articles/views.py
from itertools import chain

@login_required
def index(request):
    visits_num = request.session.get('visits', 0)
    request.session['visits'] = visits_num + 1
    request.session.modified = True
    # articles = Article.objects.all()
    # articles = Article.objects.filter(필드명__in)
    # 필드명: title, content, pk, ...
    followings = request.user.followings.all()
    # my_articles = request.user.article_set.all()
    followings_and_me = chain(followings, [request.user])
    articles = Article.objects.filter(user__in=followings_and_me)
    context = {
        'articles': articles,
        'visits': visits_num,
    }
    return render(request, 'articles/index.html', context)
```

여기서 `request.user`가 사용됐는데, 이는 로그인이 안되어있다면(`AnonymousUser`) 에러를 발생하게 됩니다. 따라서 `@login_required` decorator를 추가해 줍니다.

또한, 피드에서는 팔로잉 하는 사람들의 글 뿐만 아니라, 자신의 글 또한 확인할 수 있어야 합니다. 그런데 `my_articles`를 생성하여 list를 다루듯이 `context = {'article': articles + my_articles,}` 처럼 더해준다면 에러가 발생합니다. 왜냐하면 **queryset은 언제나 queryset 형태를 유지해야 하며, 서로 다른 메서드를 가진 queryset이 합쳐진다면 이는 그 형태를 유지할 수 없기 때문**입니다.

이를 해결하는 방법 중 하나로 **`itertools`의 `chain()`**을 사용할 수 있습니다.

