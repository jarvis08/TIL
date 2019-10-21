# Tips - If문 변형

## for문으로 if문 대신하기

### if문 사용

```html
<div class="form-group pb-2">
  {% if comments %}
  <label for="comment" class="font-weight-bold">총 {{ comments | length }}개 댓글</label>
  <table class="table table-borderless" style="table-layout:fixed">
    <thead>
      <tr>
        <th scope="col">ID</th>
        <th scope="col">내용</th>
        <th scope="col">생성일</th>
        <th scope="col">수정일</th>
        <th scope="col"></th>
      </tr>
    </thead>
    <tbody>
      {% for comment in comments reversed %}
      <tr>
        <th scope="row">{{ comment.id }}</th>
        <td>{{ comment.content }}</td>
        <td>{{ comment.created_at|date:'Y-m-d  H:i:s' }}</td>
        <td>{{ comment.updated_at|date:'Y-m-d  H:i:s' }}</td>
        <td><a href="{% url 'posts:update_comment' comment.pk %}" class="btn-sm btn-success sm">수정</a> <a href="{% url 'posts:delete_comment' post.id %}" class="btn-sm btn-danger">삭제</a></td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% else %}
    <i>아직 댓글이 없습니다. 댓글을 달아주세요 :)</i>
  {% endif %}
</div>
```

<br>

### for문의 `{% empty %}` 사용

```html
<div class="form-group pb-2">
  <label for="comment" class="font-weight-bold"><i>총 {{ comments | length }}개 댓글</i></label>
  <table class="table table-borderless" style="table-layout:fixed">
    <thead>
      <tr>
        <th scope="col">ID</th>
        <th scope="col">내용</th>
        <th scope="col">생성일</th>
        <th scope="col">수정일</th>
        <th scope="col"></th>
      </tr>
    </thead>
    <tbody>
      {% for comment in comments reversed %}
      <tr>
        <th scope="row">{{ comment.id }}</th>
        <td>{{ comment.content }}</td>
        <td>{{ comment.created_at|date:'Y-m-d  H:i:s' }}</td>
        <td>{{ comment.updated_at|date:'Y-m-d  H:i:s' }}</td>
        <td><a href="{% url 'posts:update_comment' comment.pk %}" class="btn-sm btn-success sm">수정</a> <a href="{% url 'posts:delete_comment' post.id %}" class="btn-sm btn-danger">삭제</a></td>
      </tr>
      {% empty %}
      <i>아직 댓글이 없습니다. 댓글을 달아주세요 :)</i>
      {% endfor %}
    </tbody>
  </table>
</div>
```

