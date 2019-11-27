# Heroku for Django

Heroku는 무료 서버 배포 서비스이며, 테스트 서버를 구축하는데에 용이한 서비스입니다.

```bash
$ brew tap heroku/brew && brew install heroku
$ heroku login
```

자동으로 browser를 통해 로그인 창이 열리며, 로그인 후 bash 창으로 돌아오면 자동으로 로그인이 되어 있습니다. 다음으로, django와 heroku를 연동하기 위해 다음을 설치해줍니다.

```bash
$ pip install django-heroku
```

<br>

### App 생성

아래의 명령어로 임의의 이름이 설정되는 app을 생성할 수 있습니다.

```bash
$ heroku create
```

아래와 같이 이름을 지정할 수 있으며, 자동으로 git remote로 추가됩니다.

```bash
$ heroku create jarvis-dbc
$ git remote -v
heroku	https://git.heroku.com/shrouded-headland-36935.git (fetch)
heroku	https://git.heroku.com/shrouded-headland-36935.git (push)
```

만약 remote App을 바꾸고 싶다면, 다음과 같이 변경할 수 있습니다. Heroku 웹사이트 > App > Deploy에서 추가적인 내용을 확인할 수 있습니다.

```bash
$ heroku git:remote -a 변경할app이름
```

이후 `settings.py`의 `ALLOWED_HOST`에 등록하는 주소를 변경해 주면 끝납니다. `ALLOWED_HOST`에 대한 내용은 아래에서 보다 자세히 다룹니다.

<br>

### Path 설정

Heroku 웹 페이지 > App > Settings의 `Config Vars`에 다음과 같은 `KEY` - `VALUE` 내용을 두 줄 추가합니다.

- `SCRETE_KEY` -  `TOKEN`

  `settings.py`에 설정한 `SECRETE_KEY`를 설정

- `DEBUG` - `True`

<br>

### 환경 선언

세 개의 파일을 추가하며, 내용을 작성하고, 설치를 해 줍니다.

1. `serverDirectory/runtime.txt` 파일 생성 및 내용 작성

   ```
   python-3.7.4
   ```

2. `serverDirectory/Procfile` 파일 생성 및 내용 작성

   ```
   web: gunicorn 프로젝트_이름.wsgi --log-file -
   ```

   `wsgi.py` 파일이 존재하는 경로를 위와 같이 적어준 후, `gunicorn`을 사용하기 위해 pip으로 설치해 줍니다.

   ```bash
   $ pip install gunicorn
   ```

3. `pip list` 내용(dependecies)을 heroku에게 알리기 위해 `requirements.txt` 생성

   ```bash
   $ pip freeze > requirements.txt
   ```

   `pip freeze`의 결과물을 `requirements.txt` 안에 넣음

4. 수정 내용을 heroku에 업데이트하기

   ```bash
   $ git add .
   $ git commit -m "first commit to heroku"
   $ git push heroku master
   ```

<br>

### Django Server 설정

현재 상태로 서버를 실행하려 하면, 에러가 발생합니다. 이는 장고 서버에 이 HOST를 사용함을 알리지 않았기 때문입니다. `settings.py`의 `ALLOWED_HOST`에 에러 페이지의 주소를 추가해줍니다.

<br>

### Database 설정

Heroku 서버에게 데이터베이스를 생성하도록 합니다. 만약 이미 데이터베이스가 local에 존재한다면, `__init__.py` 파일을 제외하고 migrations 디렉토리 내부 파일을 모두 삭제해야 합니다.

```bash
$ heroku run python manage.py makemigrations
$ heroku run python manage.py migrate
```

Heroku에서 사용하는 포트 번호는 `5000`입니다. 만약 사용 중인 컴퓨터의 `5000` 포트가 막혀있다면, Heroku 웹사이트의 Console을 사용하여 진행할 수 있습니다.

위의 코드와 같이 관리자 계정 또한 생성할 수 있습니다.

```bash
$ heroku run python manage.py createsuperuser
```

