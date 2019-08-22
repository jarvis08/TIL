# Model Field

- 객체 생성 시점 및 수정 시점 기록
  - `created_at = models.DateTimeField(auto_now_add=True)`

    데이터가 생성되는 시점을 기록

  - `updated_at = models.DateTimeField(auto_now=True)`

    현재 시점을 기록

