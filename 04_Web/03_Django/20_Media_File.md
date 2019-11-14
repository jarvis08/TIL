# Media File

실제로, 메인서버는 빠른 응답속도를 위해 비싼 서버 장비를 사용하며,

사진 파일과 같은 용량이 큰 데이터들은 AWS S3와 같은 파일 전용 서버에 저장한다.

<br>

<br>

## Image Upload 하기

Web URL 주소가 아닌, DB 내부의 주소를 기입하여 DB의 데이터를 활용하는 방법

`pip install Pillow`

`Shell Plus` extention과 달리 settings.py에 따로 추가하지 않아도 된다.

<br>

### 1. DB 삭제하는 작업

새로 models.py에 column을 추가해야 하므로, 번거로운 작업을 피하기 위해 DB 삭제.

1. `db.sqlite3` 파일 삭제
2. `migrations` 디렉토리의 `__init__.py` 를 제외한 파이썬 파일 삭제

<br>

### 2. Image Column 추가

```python
# models.py
class Post(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()
    # blank=True : Null 값을 허용, 부여하지 않아도 괜찮다.
    image = model.ImageField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-pk']
```

<br>

### 3. Migrate

`python manage.py makemigrations`

`python manage.py migrate`

<br>

### 4. Create 역할의 HTML 수정

   1. `form` 태그에 `enctype="multipart/form-data"` 추가

      (`enctype`을 지정하지 않았을 때의 default는 `enctype="application/x-www-form-urlencoded"`)

   2. `type="file"`인 `input` 태그를 추가

      `accept` 속성을 통해 파일 유형을 지정할 수 있는데, `image/*`을 사용하여 모든 이미지 파일 형식 수용
      
      ```html
      <input type="file" name="image" accept="image/*">
      ```
      
   3. views.py에 image 파일을 저장하도록 수정

      `request.FILES.get('image')`

      ```python
      def create(request):
          post = Post()
          post.image = request.FILES.get('image')
          post.save()
          return redirect('posts:home')
      ```

   4. settings.py에 이미지 저장 위치를 추가적으로 기입

      ```python
      # 1. 실제 파일 저장소의 경로이며, 절대경로로 기입
      MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
      
      # 2. 업로드된 파일의 주a소(URL)를 생성, 기본값은 ''
      MEDIA_URL = '/media/'
      ```

   5. `media` 또한 객체로 취급하며, `project/urls.py`에 추가적으로 기입하여 URL을 Open

      ```python
      # urls.py
      from django.contrib import admin
      from django.urls import path, include
      from posts import views
      from django.conf.urls.static import static
      # 상대경로 저장장소 불러오기
      # settings.py에 지정한 요소들 사용 가능
      from django.conf import settings
      
      urlpatterns = [
          path('admin/', admin.site.urls),
          path('', views.index, name='home'),
          path('posts/', include('posts.urls')),
      ]
      
      # /media/ 들어오는 요청을 통과시켜 주도록
      urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
      ```

   6. `detail.html`에 파일 불러오는 태그 추가

      `<img src="{{ post.image.url }}">`

      ```html
        {% if post.image %}
          <img src="{{ post.image.url }}" style="width: 100%;" alt="{{ post.image }}">
        {% else %}
          <img src="/media/no_image.png" style="width: 150px" alt="No image">
        {% endif %}
      ```

   7. `views.py`의 `update` 함수에 추가

      사진 수정 시 기존 이미지는 DB에서 삭제

      ```python
      post.image.delete()
      post.image = request.FILES.get('image')
      ```

<br>

### 5. image 저장 여부 확인해보기

```shell
# Django Shell Plus
>>> post.image
>>> post.image.url
>>> post.image.name
```