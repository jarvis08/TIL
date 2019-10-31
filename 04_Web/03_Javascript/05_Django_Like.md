# Django 좋아요 기능, JS로 동적 변화

### 기존 코드

기존 Django 코드는 좋아요 링크 클릭 시, url을 보내어 새로운 페이지를 로드했습니다. 기존 코드는 다음과 같습니다.

```html
    <h2>Like list</h2>
    <p>How many:) {{ article.like_users.count }}</p>
    <p>Who:)
      {% with likers=article.like_users.all %}
      {% for liker in likers %}
        {{ liker }}
      {% endfor %}
    </p>
    {% if user in likers %}
      <a href="{% url 'articles:like' article.pk %}" class="btn btn-secondary">Unlike</a>
    {% else %}
      <a href="{% url 'articles:like' article.pk %}" class="btn btn-danger">Like!</a>
    {% endif %}
    {% endwith %}
```

<br>

<br>

## JS 사용하기

이제는 페이지 redirect 방식이 아닌, JavaScript를 이용하여 현재 페이지를 변화시키도록 해 보겠습니다.

1. axios cdn을 base.html 상단에 작성하여 모든 js 중에서 가장 먼저 로드되도록 설정한다.

   axios는 javascript에서 url을 request하기 위해 사용한다.

   ```html
   <!-- base.html -->
   <head>
     <meta charset="UTF-8">
     <meta name="viewport" content="width=device-width, initial-scale=1.0">
     <meta http-equiv="X-UA-Compatible" content="ie=edge">
     <script src="https://unpkg.com/axios/dist/axios.min.js"></script>
     <title>BOARD</title>
   </head>
   ```

2. `Like`를 클릭하는 태그를 `a` 태그에서 **`button`** 태그로 수정합니다. `a` 태그를 사용할 경우 페이지를 로드하게 됩니다.

   ```html
   <!-- detail.html -->
   <h2>Like list</h2>
       <p>How many:) {{ article.like_users.count }}</p>
       <button id="like-button" data-id="{{ article.pk }}" class="btn btn-danger">Like</button>
   ```

   ```html
   <script>
     // 1. 좋아요 버튼을 클릭
     // 2. EventListener에 의해 axios로 view 함수 중 like에 요청(DB update)
     // 3. Like 버튼을 Unlike 버튼으로 변경
     const likeButton = document.querySelector('#like-button')
     likeButton.addEventListener('click', function(e){
       const airticleId = e.target.dataset.id
       axios.get(`/articles/${airticleId}/like/`)
           .then(response => {
             // response의 내용은 html이므로,
             if (response.data.liked) {
               e.target.classList.remove('btn-danger')
               e.target.classList.add('btn-outline-secondary')
               e.target.innerText = 'Unlike'
             } else {
               e.target.classList.remove('btn-outline-secondary')
               e.target.classList.add('btn-danger')
               e.target.innerText = 'Like!'
             }
           })
     })
   </script>
   ```

   html tag 내부에서 `data`라고 시작되는 속성 값은 모두 `dataset`에 포함되며, `button` tag에서 `data-id` 라고 설정했기 때문에 `id`라는 **key** 값이 생성됩니다. 그리고 이에 따라 `dataset.id`라고 설정하여 **value** 값인 `article.pk`를 추출할 수 있습니다.

3. 이제 **Like 여부에 따라 좋아요 아이콘을 변경해 주도록 view 함수를 수정**해 주겠습니다. 기존 view 함수 중 like를 변경해 주어야 합니다. **기존 코드**는 다음과 같습니다.

   ```python
   @login_required
   def like(request, article_pk):
       article = Article.objects.get(pk=article_pk)
       user = request.user
       # 만약 좋아요 리스트에 현재 접속중인 유저가 있다면, Unlike 처리
       if article.like_users.filter(pk=user.pk).exists():
           article.like_users.remove(user)
       else:
           article.like_users.add(user)
       return redirect(article)
   ```

   **Json 형식으로 Like 여부를 boolean 값으로 확인**할 것이므로, `JsonResponse`를 import하여 사용하겠습니다.

   ```python
   from django.http import Http404, HttpResponse, JsonResponse
   
   @login_required
   def like(request, article_pk):
       article = Article.objects.get(pk=article_pk)
       user = request.user
       if article.like_users.filter(pk=user.pk).exists():
           article.like_users.remove(user)
           liked = False
       else:
           article.like_users.add(user)
           liked = True
       # javascript로 boolean 값을 전송
       context = {
           'liked': liked,
       }
       # return redirect(article)
       return JsonResponse(context)
   ```

4. 하지만 현재 html 코드 상태로는 **페이지를 처음 불러올 때**에는 무조건 Like로 보여집니다. 따라서 detail.html 코드를 다음과 같이 수정해 줍니다.

   ```html
   {% if user in article.like_users.all %}
         <button id="like-button" data-id="{{ article.pk }}" class="btn btn-outline-secondary">Unlike</button>
       {% else %}
         <button id="like-button" data-id="{{ article.pk }}" class="btn btn-danger">Like</button>
       {% endif %}
   ```

5. **좋아요 개수를 출력**하는 것 또한 JS를 사용하여 동적으로 변경해 주도록 하겠습니다.

   ```python
   # views.py
   from django.http import Http404, HttpResponse, JsonResponse
   
   @login_required
   def like(request, article_pk):
       article = Article.objects.get(pk=article_pk)
       user = request.user
       if article.like_users.filter(pk=user.pk).exists():
           article.like_users.remove(user)
           liked = False
       else:
           article.like_users.add(user)
           liked = True
       #########################################
       context = {
           'liked': liked,
           'count': article.like_users.count(),
       }
       #########################################
       return JsonResponse(context)
   ```

   ```html
   <script>
     const likeButton = document.querySelector('#like-button')
     likeButton.addEventListener('click', function(e){
       const airticleId = e.target.dataset.id
       axios.get(`/articles/${airticleId}/like/`)
           .then(response => {
   ///////////////////////////////////////////////////////////
             document.querySelector('#like-count').innerText = response.data.count
   ///////////////////////////////////////////////////////////
             if (response.data.liked) {
               e.target.classList.remove('btn-danger')
               e.target.classList.add('btn-outline-secondary')
               e.target.innerText = 'Unlike'
             } else {
               e.target.classList.remove('btn-outline-secondary')
               e.target.classList.add('btn-danger')
               e.target.innerText = 'Like!'
             }
           })
     })
   </script>
   ```

