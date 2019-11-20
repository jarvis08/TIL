# Django DRF, API 활용

### Log-in 처리 과정

1. ID/PW 입력 후 `Log-in` 버튼 클릭
2. Vue의 `axios`를 통해 `http://127.0.0.1:8000/api-token-auth/` 경로로 ID/PW를 전송
3. 정보가 확인되면, 서버가 token을 return
4. Vue의 `.then`을 통해 vue-session을 열고, token인 JWT를 `jwt`라는 이름의 key로 저장
5. `router`를 이용하여 Home.vue을 의미하는 `'/'`로 이동

<br>

### Log-in 이후

JWT를 Header에, 필요 정보를 Body에, 요구되는 요청 방식(POST, GET 등)으로, 목적하는 url을 통해 request하면 json 형태의 데이터가 return 됩니다.

<br>

<br>

## To Do List API 구현하기

### admin으로 이전 todo list 확인

```python
# todos/admin.py
from django.contrib import admin
from .models import Todo

# Register your models here.
admin.site.register(Todo)
```

설정 후 `localhost:8000/admin/`에 접속하면 Postman을 통해 생성한 todo 데이터를 확인할 수 있습니다.

<br>

### 유저 정보 요청하기

```python
# todos/urls.py
from django.urls import path
from . import views
urlpatterns = [
    path('todos/', views.todo_create),
    path('user/<int:pk>/', views.user_detail),
]
```

User 정보를 검증하기 위한 UserSerializer를 작성합니다.

```python
# todos/serializers.py
from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Todo

class TodoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Todo
        fields = ('id', 'user', 'title', 'completed', )


class UserSerializer(serializers.ModelSerializer):
    todo_set = TodoSerializer(many=True)
    class Meta:
        model = get_user_model()
        fields = ('id', 'username', 'todo_set', )
```

```python
# todos/views.py
from django.shortcuts import render, get_object_or_404
from .serializers import TodoSerializer, UserSerializer
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.contrib.auth import get_user_model


@api_view(['POST'])
def todo_create(request):
    serializer = TodoSerializer(data=request.POST)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    return Response(status=400)


@api_view(['GET'])
def user_detail(request, pk):
    User = get_user_model()
    user = get_object_or_404(User, pk=pk)
    # JWT의 user 정보 == User.pk 일 경우에만 detail을 반환
    if request.user != user:
        return Response(status=404)
    serializer = UserSerializer(user)
    return Response(serializer.data)
```

```json
// Postman으로 return 받은 json data
{
    "id": 1,
    "username": "admin",
    "todo_set": [
        {
            "id": 1,
            "user": 1,
            "title": "3",
            "completed": false
        },
        {
            "id": 2,
            "user": 1,
            "title": "3",
            "completed": false
        }
    ]
}
```

<br>

### todo 목록 가져오기

우선 return 받은 jwt 파일을 decode하여 유저 정보(user_id, user_name 등)를 parsing해야 하므로, decoding을 위한 라이브러리를 설치하겠습니다.

```bash
$ npm i jwt-decode
```

```vue
<!-- Home.vue -->
<template>
  <div class="home">
    <h1>Todo w/ Django & Vue</h1>
    <TodoInput />
    <TodoList :todos="todos"/>

  </div>
</template>

<script>
// @ is an alias to /src
import TodoList from '@/components/TodoList.vue'
import TodoInput from '@/components/TodoInput.vue'
import router from '@/router'
import jwtDecode from 'jwt-decode'
import axios from 'axios'

export default {
  name: 'home',
  data() {
    return {
      todos: []
    }
  },
  components: {
    TodoList,
    TodoInput,
  },
  methods: {
    // Page load 마다 자동으로 불려야 하므로 mounted()에 등록하며,
    // 로그인 여부를 확인하여 비로그인 시 /login으로 강제 이동
    loggedIn() {
      this.$session.start()
        // jwt 라는 이름의 key가 없다면, 비로그인이 상태라면
        if (!this.$session.has('jwt')) {
          router.push('/login')
          // login으로 가라
        }
    },
    // Log-in처럼, page load 시 마다 자동으로 불려야 하는 함수이므로 mounted()에 추가
    getTodos() {
      const token = this.$session.get('jwt')
      const user_id = jwtDecode(token).user_id
      const options = {
        headers: {
          Authorization: 'JWT ' + token
        }
      }
      axios.get(`http://localhost:8000/api/v1/user/${user_id}/`, options)
      .then(res => {
        this.todos = res.data.todo_set
      })
    },
  },
  // 8개의 life cycle hook
  // Page가 load될 때 마다, 자동으로 매번 불려야 함수들을 지정
  mounted() {
    this.loggedIn()
    this.getTodos()
  },
}
</script>
```

<br>

### todo 추가하기

```vue
<!-- TodoInput.vue -->
<template>
  <div class="todo-input">
    <h2>New to work</h2>
    <!-- Enter 혹은 button click 모두 사용 가능하도록 form tag 사용 -->
    <!-- 원래 기능은 없애고, 뒤에 정의한 createTodo method를 실행하도록 처리 -->
    <form class="input-group mb-3" @submit.prevent="createTodo">
      <input v-model="title" type="text" class="form-control">
      <button type="submit" class="btn btn-primary">+</button>
    </form>
  </div>
</template>

<script>
export default {
  name: 'TodoInput',
  data() {
    return {
      title: ''
    }
  },
  methods: {
    createTodo() {
      // emit을 할 뿐, 직접 axios로 todo를 생성하는 것은 아니며,
      // todo_set에 추가하는 작업은 Home.vue에서 진행
      // event 발생임을 Home에서 편리하게 보기 위해 Event라고 명시
      this.$emit('createTodoEvent', this.title)
      // input 초기화
      this.title = ''
    },
  }
}
</script>
```

```vue
<!-- Home.vue -->
<template>
  <div class="home">
    <h1>Todo w/ Django & Vue</h1>
    <!-- TodoInput으로 부터 createTodoEvent 발생 시 createTodo 실행 -->
    <TodoInput @createTodoEvent="createTodo" />
    <TodoList :todos="todos"/>

  </div>
</template>

<script>
// @ is an alias to /src
import TodoList from '@/components/TodoList.vue'
import TodoInput from '@/components/TodoInput.vue'
import router from '@/router'
import jwtDecode from 'jwt-decode'
import axios from 'axios'

export default {
  name: 'home',
  data() {
    return {
      todos: []
    }
  },
  components: {
    TodoList,
    TodoInput,
  },
  methods: {
    // Page load 마다 자동으로 불려야 하므로 mounted()에 등록하며,
    // 로그인 여부를 확인하여 비로그인 시 '/login'으로 강제 이동
    loggedIn() {
      this.$session.start()
        // jwt 라는 이름의 key가 없다면, 비로그인이 상태라면
        if (!this.$session.has('jwt')) {
          router.push('/login')
          // login으로 가라
        }
    },
    // Log-in처럼, page load 시 마다 자동으로 불려야 하는 함수이므로 mounted()에 추가
    getTodos() {
      const token = this.$session.get('jwt')
      const user_id = jwtDecode(token).user_id
      const options = {
        headers: {
          Authorization: 'JWT ' + token
        }
      }
      axios.get(`http://localhost:8000/api/v1/user/${user_id}/`, options)
      .then(res => {
        this.todos = res.data.todo_set
      })
    },
    createTodo(title) {
      const token = this.$session.get('jwt')
      const user_id = jwtDecode(token).user_id
      const options = {
        headers: {
          Authorization: 'JWT ' + token
        }
      }
      // const data = {}의 객체 형태도 가능하지만, FormData() 객체 사용
      const data = new FormData()
      // data.append(key, value)
      // data.append(input_name, input_value)
      data.append('user', user_id)
      data.append('title', title)
      axios.post('http://localhost:8000/api/v1/todos/', data, options)
      .then(res => {
        this.todos.push(res.data)
      })
    }
  },
  // 8개의 life cycle hook
  // Page가 load될 때 마다, 자동으로 매번 불려야 함수들을 지정
  mounted() {
    this.loggedIn()
    this.getTodos()
  },
}
</script>
```

`FormData()` [참고자료](https://developer.mozilla.org/ko/docs/Web/API/FormData)

`mounted()` [참고자료](https://beomy.tistory.com/47)

<br>

### Server와 Front를 분리하는 이유

서버는 그대로 유지한 채 mobile application을 제작하여 연동할 수 있으며, React.js의 경우 프론트의 코드를 그대로 mobile application으로 사용할 수 있습니다. 따라서 Django만을 이용하는 것 보다 SPA를 활용하는 것이 더 확장성이 좋습니다.

























