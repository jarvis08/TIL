# Tips - Counting

## 개수 표시하기

1. 파이썬에서 연산하여 보내므로 SQL문이 하나 추가됨, 비효율적

   ```html
   총 {{ comments.count }}개의 댓글
   ```

2. DTL 사용, 추천하는 방법

   ```html
   총 {{ comments | length }}개의 댓글
   ```

<br>