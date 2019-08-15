# Display

---

## Block

- 항상 새로운 라인에서 시작

- 화면 크기 전체의 가로폭을 차지(width: 100%)

- block 레벨 요소 내에 inline 레벨 요소를 포함 가능

  block 레벨은 수가 많으니, inline 레벨 요소를 암기할 것!

  - block level element : `div`, `h1~h6`, `p`, `ol`, `ul`, `li`, `hr`, `table`, `form`
  - `display: inline;` 추가 시 모두 무시하고 inline으로 취급

  ```css
  div {
    /*margin-left/right 모두 auto로 하면 가운데 정렬 효과*/
    margin-left: auto;
    margin-right: auto;
    height: 100px;
    width: 300px;
  }
  ```

---

## Inline

- 문장 삽입 가능

- content의 너비만큼 가로폭을 차지

  따라서 font size가 커지면 inline의 width 증가

- `height`, `width`, `margin-top/bottom` 등의 property 지정 불가!

  - `margin,padding-left/right` property 지정 가능
  - 하지만 `auto` 지정은 불가

- 상/하 여백은 `line-height`로 지정

- Inline level element

  - `<a>`,` <span>`, `<img>`, `<br>`, `<strong>`

  - `<form>` 내부에서 사용되는 tag들

    `<input>`, `<select>`, `<textarea>`, `<button>`

---

## Inline-Block

- inline level element 처럼 한 줄에 표시되지만,

  block에서의 width, height, margin(top, bottom) property를 모두 지정 가능

  ```html
  <div class="inline-block">inline-block<span>붙여쓰기</span></div>
  ```

  ```css
  .inline-block {
    display: inline-block;
    margin: 50px;
    height: 100px;
    width: 100px;
  }
  ```

---

## None

- 해당 요소를 화면에 표시하지 않으며, 공간 조차 사라지게 조치

- 동적으로 작업할 때 사용

- 공간은 유지하고 싶을 때, `visibility`를 사용

  *e.g., log-in 시 log-in 버튼 사라지게 하기*

```html
<div class="none">얘는 곧 사라집니다.</div>

.none {
    display: None;
}
```

---

## Float

- 정렬하기 위해 사용되며,

  객체를 오른쪽/왼쪽으로 정렬하여 문서 배치(layout)를 조정

- 묶음 태그에만 적용 가능한 속성

  `<div>`, `<p>`, `<ol>`, `<ul>`, `<table>`, `<img>` 등

- 속성값

  `left`, `right`, `none`

- 위로 띄워서 이동

---

## Flex

> Bootstrap의 Grid System 또한 flex 개념

- grid는 flex에 heavy하게 의존하는 형태

- flex는 전체 html로부터 해당 contents를 box 형태로 따로 관리할 수 있도록 분리

- `display: flex;`를 지정하면, 박스에 지정한 `height`의 길이만큼 height 부여

  ```css
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

- `align-items: 정렬형태`

  - `justify-content`와 `align-items`는 서로 반대의 세로/가로 축의 정렬을 관리하며,

    각각의 축은 `flex-direction`에 의해 결정

    e.g., `justify-content`가 세로일 경우 `align-items`가 가로, 혹은 그 역순

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

- `justify-content`

  flexbox 전체의 정렬 형태를 결정

  - `justify-content: center/flex-start/flex-end;`
    - `center` : 중앙 정렬
    - `flex-start` : 좌측 시작
    - `flex-end` : 우측 시작
  - `justify-content: space-between/space-around;`
    - `space-between` : 컨텐츠 사이가 동일한 공간을 갖는 것에 집중
    - `space-around` : 양쪽 끝까지 공간을 동일하게 분할

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

---

## Grid

- 격자 형식으로, Web Page의 구간을 나누어 규격을 맞추는 방법

- 가장 먼저는 가로(행) 배열을 고려

  그 이후 세로 정렬 형태 고려

- 현재는 대부분 표준화된 Grid(12개) 사용

  12개 Grid 분할은 2, 3, 4, 6개의 column으로 나누어 post를 띄움

  (약수가 많기 때문에 12가 표준으로 지정됨)

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