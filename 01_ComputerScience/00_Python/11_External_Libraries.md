# Python External Libraries

---

## decouple

:: key 암호화시키기

- directory 안에 `.env` 파일 생성(linux에서는 .으로 시작하면 숨김파일)

- 모두 대문자로 작성

```python
# .env 파일 내부에 아래와 같이 작성하며, .env파일은 공유되지 않아야한다.
TELEGRAM_TOKEN = "토큰 정보 기입"
# 이를 위해 .gitignore 파일을 생성하고, 무시하고자 하는 파일명을 기입
.env
```

```python
# token url 사용할 때
from ducouple import config
# token_url을 원래 토큰 값 대신 config('.env에 작성한 token을 넣은 변수명')
token_url = config('TELEGRAM_TOKEN')
```

- `Linux` 에서는 환경변수 안에 저장하여, 언어에 무관하게 암호화된 키를 사용 가능

---

