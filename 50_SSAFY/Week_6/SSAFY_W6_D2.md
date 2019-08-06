# SSAFY Week6 Day2

---

- `<label>` tag를 통해 `<input>` tag가 어떤 input을 받는지 설명해 주며, semantic web을 위해 필수적인 tag

  `<label>` 의 `for`와 `<input>`의 `id`는 항상 동일해야 한다!

  ```html
  <label for="exampleInputEmail1">Email address</label>
  <input type="email" class="form-control" id="exampleInputEmail1" aria-describedby="emailHelp" placeholder="Enter email">
  ```

- `class="justify-content-center"`

  반응형 제작 후, mobile 버전으로 변경됐을 때 중앙 정렬 하여 contents를 보여주는 class

  ```html
  <div class="row m4-4 justify-content-center">
      <div class="col-sm-4 col-12 card" style="width: 18rem;">
          <img src="https://images8.alphacoders.com/282/thumb-1920-282535.jpg" class="card-img-top" alt="spiderman">
          <div class="card-body">
              <h5 class="card-title">Ironman</h5>
              <p class="card-text">아이언맨을 못 보는게 아쉽다.</p>
              <a href="#" class="btn btn-primary">Visit Shield</a>
          </div>
      </div>
  </div>
  ```

- `col-12`를 카드 별로 추가적으로 작성 시, mobile 버전에서 전체 가로 grid를 사용

  ```html
  <div class="col-sm-4 col-12 card" style="width: 18rem;">
      <img src="https://images8.alphacoders.com/282/thumb-1920-282535.jpg" class="card-img-top" alt="spiderman">
      <div class="card-body">
          <h5 class="card-title">Ironman</h5>
          <p class="card-text">아이언맨을 못 보는게 아쉽다.</p>
          <a href="#" class="btn btn-primary">Visit Shield</a>
      </div>
  </div>
  ```

---

## Display, Flex

> Bootstrap의 Grid System 또한 flex 개념

- grid는 flex에 heavy하게 의존하는 형태

- flex는 전체 html로부터 해당 contents를 box 형태로 따로 관리할 수 있도록 분리

- `display: flex;`를 지정하면, 박스에 지정한 `height`의 길이만큼 height 부여

  ```html
    <style>
    .container {
      height: 800px;
      padding: 16px;
      border: 2px solid black;
      display: flex;
    }
    .item {
      padding: 16px;
      border: 2px solid black;
      font-weight: bold;
    }
    </style>
  
  <body>
    <div class="container">
      <!-- 길이가 800px인 세로가 긴 줄 5개 생성하여 횡(가로, row) 정렬 -->
      <!-- flex가 아니라면, 원래 box 형태인 가로 줄 5개 생성 -->
      <div class="item">1</div>
      <div class="item">2</div>
      <div class="item">3</div>
      <div class="item">4</div>
      <div class="item">5</div>
    </div>
  ```

- `flex-direction: row/row reverse/column/column revers;`

  - `row`

    default 값이며, 횡 정렬

    bootstrap의 `container` > `row` class는 `flexbox`의 `row`와 동일

  - `row reverse`

    역순으로 screen의 오른쪽에 횡 정렬

  - `column`

     평범한 가로 정렬 (1, 2, 3, 4, 5)

  - `column reverse`

    역순 가로 정렬 (5, 4, 3, 2, 1)

- `justify-content`

  flexbox 전체의 정렬 형태를 결정

  - `justify-content: center/flex-start/flex-end;`
    - `center` : 중앙 정렬
    - `flex-start` : 좌측 시작
    - `flex-end` : 우측 시작
  - `justify-content: space-between/space-around;`
    - `space-between` : 컨텐츠 사이가 동일한 공간을 갖는 것에 집중
    - `space-around` : 양쪽 끝까지 공간을 동일하게 분할

- `align-items: 정렬형태`

  - `justify-content`와 `align-items`는 서로 반대의 세로/가로 축의 정렬을 관리하며,

    각각의 축은 `flex-direction`에 의해 결정

    i.g., `justify-content`가 세로일 경우 `align-items`가 가로, 혹은 그 역순

  - 가능한 속성 값은 `justify-content`와 동일하며, default 값은 `flex-start`

- `flex-wrap: nowrap/wrap/wrap-reverse;`

  - `nowrap;`이 default이기 때문에 contents 개수/`width`/`height`에 따라 container를 벗어날 수 있다.

  - `wrap;`을 부여하여 자동으로 넘친 내용에 대해 복수의 line으로 분리

    넘쳐서 분리된 line에도 지정한 `justify-content` 내용을 적용

  - 반응형의 경우 이 과정이 더욱 복잡해지기 때문에

    contents의 size를 결정하는 것이 front-end dev에게 중요한 요인

- `flex-flow: row wrap`

  `flex-direction`과 `flex-wrap`을 동시에 사용 가능

- content 다루기

  - `order: 1`;

    해당 content를 -1일 시 맨 앞으로, +1일 시 맨 뒤로 보냄

  - `align-self: flex-end;`

    자신 하나만의 위치를 이동(속성 값은 동일하게 사용 가능)

  - `align-content: flex-start;`

    여러 줄의 contents 사이 간격을 지정

---

## Display, Grid

- `display: grid;`

  - grid 설정

    ```css
    #garden {
      display: grid;
      grid-template-columns: 20% 20% 20% 20% 20%;
      grid-template-rows: 20% 20% 20% 20% 20%;
    }
    ```

    - `grid-template-columns: repeat(n, %);`

      지정 %로 n개의 구간 생성

    - `grid-template-columns: px em %;`

      여러 단위로 지정 가능

  - column 구간 지정

    - `grid-column-start: 3;`

      `grid-column-end: 3;`

      - 음수 값 가능(python index처럼 사용)

      - `grid-column-start: span 3;`

        `grid-column-end: span 3;`

        넓이로 지정 가능

    - `grid-column: 4 / 6;`

      /를 이용하여 동시 지정 가능

      - `grid-column: 2 / span 3`

        2부터 시작하여, 넓이 3

  - row 구간 지정

    column 지정과 동일

  - row / column 동시 구간 지정

    `grid-area: row-start / column-start / row-end / column-end;`

  ```css
  #water {
  /* grid column 3,4 지정 */
  /* start: 4; end: 3;으로 지정해도 같은 결과 */
  /* 음수 값 가능하며, python index처럼 사용 */
  grid-column-start: 3;
  grid-column-end: 5;
  }
  ```

---

## Media Query

- `viewport`

  - webpage에서 보이는화면의 크기이자 기준 폭

  - apple에서 처음 개발

  - pc의 경우 browser 크기
  - `content`
    - `width`와 `initial-scale`을 정의
    - `device-width` 설정을 위해 만들어진 속성
    - `initial-scale=1.0` : 1배율로 시작함을 의미

  ```html
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
  </head>
  ```

- media query

  `@media (조건) { 내용 }`

  viewport를 통해 받은 기기에 대한 정보를 이용하여,

  중재하는 media에 대해 조건적으로 작업을 수행 

  특정 조건을 적용할 때 `@대상 + (조건){ 내용 }` 를 사용

  ```css
  <style>
      h1 {
          color: red;
      }
      /* with가 1024px이하 일 때 */
      @media (max-width: 1580px) {
          h1 {
              color: blueviolet;
          }
      }
      /* 대체로 max 보다는 min을 사용하며, '지정 값보다 클 때'에 적용 */
      @media (min-width: 500px) {
          h1 {
              display: none;
              color: darksalmon;
          }
      }
  </style>
  ```

  > 주로 사용되는 조건
  >
  > `min-width: 576px`
  >
  > `min-width: 768px`
- `ot` : device 방향(세로, 가로 모드)

---

## Animate.css

- animation 효과 부여

  https://daneden.github.io/animate.css/

  ```html
  <!-- CDN -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/3.7.2/animate.min.css">
  ```

---

## Font Awesome

- github, facebook 같은 icon 배포

  https://fontawesome.com/

  ```html
  <!-- CDN -->
  <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.10.1/css/all.css" integrity="sha384-wxqG4glGB3nlqX0bi23nmgwCSjWIW13BdLUEYC4VIMehfbcro/ATkyDsF/AbIOVe" crossorigin="anonymous">
  ```

- size 조절

  https://fontawesome.com/how-to-use/on-the-web/styling/sizing-icons

  ```html
  <i class="fas fa-camera fa-xs"></i>
  <i class="fas fa-camera fa-sm"></i>
  <i class="fas fa-camera fa-lg"></i>
  <i class="fas fa-camera fa-2x"></i>
  <i class="fas fa-camera fa-3x"></i>
  <i class="fas fa-camera fa-5x"></i>
  <i class="fas fa-camera fa-7x"></i>
  <i class="fas fa-camera fa-10x"></i>
  ```

---

## Codepen

- God Designers' open source

  https://codepen.io/popular/pens/