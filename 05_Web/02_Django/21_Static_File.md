# Static File

Static File : 고정된 위치에 저장해둔 후 불러와서 사용하는 파일

i.g., Favicon(16x16 image), Open Graph

오픈 그래프 : 카카오톡 등의 채팅방에서 링크 기입 시 사진과 같은 형태로 보여지는 미리보기 및 로고

<br>

<br>

### Image 사용하기

1. `root경로/static/App_Name/image.png`  경로에 파일 저장

2. `settings.py` 수정

   ```python
   # settings.py
   STATICFILES_DIRS = [
       os.path.join(BASE_DIR, "static"),
       '/var/www/static/',
   ]
   ```

3. `Pjt_Name/urls.py` 수정

   ```python
   # PjtName/urls.py
   from django.conf.urls.static import static
   from django.conf import settings
   urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
   ```

4. 이미지를 업로드하고자 하는 html 파일 수정

   ```html
   <!-- html file -->
   {% load static %}
   <img src="{% static 'movies/I_also_have_purity.jpg' %}" style="width: 100%;">
   ```

<br>

<br>

## Favicon 생성

- 무료 파비콘 생성

  [Favicon Generator](https://www.favicon-generator.org/)

settings.py의 `STATIC_URL = '/static/'`을 수정하여 경로를 수정할 수 있지만, 유지하는 것이 Convention

<br>

### Django's Convention

1. settings.py의 내용은 유지

2. `AppName/static/AppName/fabicon.jpg` 형태로 파일 위치

3. html 파일에는 다음과 같이 명시

   ```html
   {% load static %}
   <!doctype html>
   <html lang="en">
   <head>
       <link rel="icon" type="image/png" sizes="16x16" href="{% static 'AppName/fabicon-16x16.png' %}">
   </head>
   ```

<br>

### 전역으로 관리하기

templates 디렉토리를 project 디렉토리로 가져와서 모든 앱에 사용할 공통 template을 한꺼번에 관리하는 것과 같은 개념

1. settings.py에 전역 폴더 지시

   ```python
   # Static files (CSS, JavaScript, Images)
   # https://docs.djangoproject.com/en/2.2/howto/static-files/
   # appname/static/스태틱파일 위치
   STATIC_URL = '/static/'
   
   # 전역 탐색 디렉토리 만들기
   # 이러한 내용을 assets라고 표현하는 것이 convention
   # STATICFILES_DIRS = [os.path.join(BASE_DIR, 'projectName', 'assets')]
   STATICFILES_DIRS = [os.path.join(BASE_DIR, os.path.basename(os.getcwd()), 'assets')]
   ```

2. `project_name` 디렉토리 안에 `assets` 디렉토리 안에 `images` 디렉토리 생성

   html파일 수정

   ```html
   {% load static %}
   <!doctype html>
   <html lang="en">
   <head>
       <link rel="icon" type="image/png" sizes="16x16" href="{% static '/images/fabicon-16x16.png' %}">
   </head>
   ```

   