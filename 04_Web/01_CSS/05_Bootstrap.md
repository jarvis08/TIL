# Bootstrap

- Twitter에서 개발

  https://getbootsrap.com

- designspectrum.org

  design의 표준을 알아볼 수 있는 website

- Bootstrap은 css가 아닌 scss를 사용하여 css 파일 제작

  따라서 source file은 scss로 작성되어 있으며, compiled 파일을 받아야 확인 가능

- Bootstrap Quick Start

  CDN(Content Delivery Network)을 이용하여 효율적인 컨텐츠(CSS, JS, Image, Text 등)를 배포

  여러 노드에 가진 네트워크에 데이터를 제공하는 시스템

  - e.g., Amazon Cloud Front

  > 개별 end-user의 가까운 서버를 통해 빠르게 전달
  >
  > 외부 서버를 활용함으로써 본인 서버의 부하 감소
  >
  > 보통 적절한 수준의 캐시 설정으로 빠르게 로딩 가능
  >
  > 캐시 : 사용한 경험이 있는 것은 저장하여, 더 빠르게 로딩

### Utilities

편하게 클래스로 적용

<br>

### Spacing

기존의 `margin` 조작 대신, 편하게 공간 확보

`rem`으로 값을 배정하며, browser default가 16px 정도이므로 대략적인 계산 가능

```html
<!-- 기존 -->
<style>
    h1 {
        margin-top: 10px;
    }
</style>
```

```html
<!-- Bootstrap을 사용한 html에서의 class 설정 -->
<!-- m-5 : margin value 5(48px) -->
<!-- mx, mt, pt, pb 등 margin과 padding 설정 가능 -->
<h1 class="mt-1">편리한 margin 부여</h1>
```

```css
/* bootstrap.css */
.m-0 {
  margin: 0 !important;
}
/* important를 부가하여 우선순위 배정 */
```

- `mt-1` = 0.25 `rem`

  `mt-2` = 0.5 `rem`

  `mt-3` = 1 `rem`

  `mt-4` = 1.5 `rem`

  `mt-5` = 3 `rem`

  1.5 rem부터는 크다고 느껴지며, 보통 2혹은 3 정도를 적당하다고 판단

- `m`, `mt`, `mb`, `ml`, `mr`, `mx`, `my`, `m-n1~4`, `mx-autio`

  = 전체, top, bottom, left, right, x축, y축, `-(negative)`

- mx-auto, ml-auto, mr-auto의 경우 잘 쓰이지 않는다.

  주로  `position: flex`와 `grid`를 이용하여 설정

<br>

### Color

| Purpose and Name | Hash Value |
| ---------------- | ---------- |
| primary          | #007bff    |
| secondary        | #6c757d    |
| success          | #28a745    |
| info             | #17a2b8    |
| warning          | #ffc107    |
| danger           | #dc3545    |
| light            | #f8f9fa    |
| dark             | #343a40    |

- 사용법

  - `class="bg-primary"`

  - `class="text-primary"`

  - `{color: primary;}`

  - `class="alert-primary"`

    배경색과 텍스트(동일 계열) 색을 변경

  - `class="btn-primary"`

    버튼 색 변경

    ```html
    <button class="btn btn-primary">버튼 1</button>
    <button class="btn-primary">버튼 2</button>
    <a class="btn btn-dark" href="">이쁜 링크</a>
    ```

<br>

### Border

- `class="border border-success rounded"`

  ```html
    <div class="border border-success rounded">
      <p>보더 테스트</p>
    </div>
  ```

<br>

### Display

- 기존 : block, inline-block, inline, none

- `class="d-block"`

  ```html
  <div class="border border-success rounded">
      <p class="d-inline">보더 테스트</p>
  </div>
  ```

  - 반응형 맛보기

    Device 크기에 따라 sm(Small), md(Medium), lg(Large), xl(Extra Large)

    Browser 창 크기 조절하여 확인 가능

    Extra Small은 default 값으로 사용

    - `class="d-sm-none"` : Mobile Screen
    - `class="d-md-none"` : Tablet 세로 Screen
    - `class="d-lg-none"` : 정사각형 Screen
    - `class="d-xl-none"` : Wide Screen

<br>

### Position

`class="position-static/relative/absolute/fixed/float"`

- `class="fixed-top/bottom"`

- Text

  https://getbootstrap.com/docs/4.3/utilities/borders/

  공식 홈페이지 Unitilities에서 자세하게 확인 가능

  - `class="text-center"`

    `class="font-weight-bold"`

    `class="font-italic"`

  - **color와 breakpoint는 공식 홈페이지의 Components에서 자세히 확인 가능**

    https://getbootstrap.com/docs/4.3/components/alerts/

- 자주 사용 할 Components

   - Card

   - Carousel(회전목마)

     화살표로 넘기는 Auto Slide

   - Collapse

     접어서 내용 비가시 효과

   - Dropdowns

     Mobile이 사용시 Button에서의 에러 발생이 잦다.

   - **Forms**

     무엇을 하든 사용자로부터 입력값을 받아야 하기 때문에 자주 사용

   - Input group

   - Jumbotron

     전광판 효과

   - List group

     여러 게시판들 사이를 자유롭게 오갈 수 있도록 도움

   - Modal

     팝업창 효과, Mobile에서 깨지기 쉬워 자주 사용하지는 않는다.

   - Navs, Navbar

   - Pagination

     페이지 구분하여 만들 때

   - Popovers

     팝업보다는 규모가 적게 살짝 뜨는 것

     (마우스 가져다대면 설명 보여주기 등)

   - Progress

     진행 bar 표시

   - Scrollspy

- `rebootstrap.css`

  open source인 `normalize.css` 내용을 포함

  html 설정을 설정한 default로 바꿔주는 역할

- margin collapsion을 제거하기 위해

  bootstrap에서는 기본적으로 윗 마진을 제거

- javascript의 script 파일을 `<body>`에서 선언하는 이유

  `<head>`에 있을 시 heavy해지는 성능상(속도)의 제한으로 body에 위치

