# V, VM, M

## View(V)

사용자에게 보여지는 부분을 말하며, html 요소들을 지칭합니다.

- `<head> 내용 </head>`
- `<body> 내용 </body>`

<br>

### Vue Directive

```html
<p v-for></p>
<p v-if v-else v-else-if></p>
<p v-model></p>
<p v-on:이벤트></p>
<p @:이벤트></p>
<p v-bind:html속성이름></p>
<p :html속성이름></p>
<p v-html></p>
<p v-text></p>
<p v-show></p>
```

<br>

<br>

## ViewModel(VM)

ViewModel에서는 **Vue Instance**를 사용하여 데이터를 **업데이트**시키고, 보여줍니다.

`<script> ViewModel 정의 </script>`

<br>

### Arrow Function을 사용하지 않는 요소들

1. `EventListener(함수)`
2. (Vue Instance > `methods` property)에서의 method 선언

<br>

### Properties

Vue Instance의 properties에는 다음의 요소들이 있습니다.

- `el: '#id'`

  **View(html)**의 **어떤 요소**에 **Vue Instance**(ViewModel)를 **mount**할 지 지정합니다.

- `data: {변수: 값,}`

  Vue Instance가 사용할 Data입니다.

  ```javascript
  data: {
    자료이름(identifier): 값,
    자료이름(identifier): 배열,
  }
  ```

- `methods: {}`

  Vue Instance가 사용할 메소드들을 정의하는 곳입니다.

  ```vue
  methods: {
    함수명: function(인자) {내용},
    함수명(인자) {내용},
    함수명(인자) {내용},
  }
  ```

- `computed: {}`

  미리 계산되어 있는, **캐싱된 값을 반환**합니다. 이는 성능 상의 이유로 사용되는 경우가 많습니다.

- `watch: {}`

  Vue 인스턴스의 **data 변경을 관찰**하고, 이에 **반응**합니다.

  ```javascript
  watch: {
    지켜보는data: {
      // handler method: 지켜보는 데이터가 변경되면 실행하는 함수
      // 함수명 또한 handler 라고 사용
      handler(지켜보는data) {
        함수 정의
      },
      // deep copy를 할 것인지?
      deep: true,
    }
  }
  ```


<br>

<br>

## Model

데이터를 관리합니다.



























