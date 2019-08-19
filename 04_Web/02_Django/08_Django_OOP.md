# OOP in Django

- 객체(class) 생성하여 template에 넘기기

  `created_at`, `title`, `content`

  ```python
# views.py
  blogs = []
  class Article:
      def __init__(self, title, content, created_at):
          self.title = title
          self.content = content
          self.created_at = created_at
      
      def __str__(self):
          return f'제목: {self.title}, 내용: {self.content}, 작성시간: {self.created_at}'
  
  def index(request):
      context = {
          'blogs': blogs,
      }
      return render(request, 'index.html', context)
  
  def create(request):
      created_at = datetime.now()
      title = request.GET.get('title')
      content = request.GET.get('content')
  
      # blogs.append({'title': title, 'content': content, 'created_at': created_at})
      # 객체로 만들기
      blogs.append(Article(title, content, created_at))
  ```
  
  ```html
<!-- index.html -->
    {% for blog in blogs %}
      <p>제목: {{ blog.title }}</p>
      <p>내용: {{ blog.content }}</p>
      <p>작성시간: {{ blog.created_at }}</p>
  	<p>__str__: {{ blog }}</p>
    {% endfor %}
  ```
  
  ```
# 결과
  제목: 첫
  내용: 첫내용
  작성시간: Aug. 19, 2019, 11:34 a.m.
  __str__: 제목: 첫, 내용: 첫내용, 작성시간: 2019-08-19 11:34:52.854799
  제목: 둘
  내용: 두번째내용
  작성시간: Aug. 19, 2019, 11:34 a.m.
  __str__: 제목: 둘, 내용: 두번째내용, 작성시간: 2019-08-19 11:34:59.982262
  ```