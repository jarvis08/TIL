# GET + POST

## new/create 합치기

현재 각종 `form`에서는 `method="POST"`를 사용하고 있다. 그리고 html 파일을 가져오는 것은 일반적으로 `GET` 방식으로 사용된다. 따라서 이를 이용하여 `request.method == 'GET'`일 시 기존의 `new` 혹은 `edit`의 내용을 호출하며, 그 외의 경우(`POST`)에는 기존 `create`와 `update`의 내용을 호출하는 방식으로 view 함수를 수정할 수 있다.

- `new`/`create`

  ```python
  # create와 new 합치기
  # def new(request):
  #     return render(request, 'todos/new.html')
  
  
  def create(request):
      if request.method == 'POST':
          title = request.POST.get('title')
          due_date = request.POST.get('due-date')
          Todo.objects.create(title=title, due_date=due_date)
          return redirect('todos:index')
      else:
          return render(request, 'todos/create.html')
  ```

<br>

### edit/update 합치기

- `edit`/`create`

  ```python
  # def edit(request, pk):
  #     todo = Todo.objects.get(pk=pk)
  #     context = {
  #         'todo': todo,
  #     }
  #     return render(request, 'todos/edit.html', context)
  
  
  def update(request, pk):
      todo = Todo.objects.get(pk=pk)
      if request.method == 'POST':
          todo.title = request.POST.get('title')
          todo.due_date = request.POST.get('due-date')
          todo.save()
          return redirect('todos:index')
      else:
          context = {
              'todo': todo,
          }
          return render(request, 'todos/update.html', context)
  ```


