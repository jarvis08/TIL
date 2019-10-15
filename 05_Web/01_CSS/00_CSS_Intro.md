# Cascading Style Sheets

참고 자료 : ./10_Lecture_Scripts/03_css.pdf

<br>

<br>

## CSS Intro

- HTML, Hyper Text Markup Language

  - Markup Language, Content

  - 정보 + 구조화

- CSS, Cascading Style Sheet

  - Style Sheet Language, Presentation

  - Styling을 정의

- 기본 사용법

  `color: 이름 / #hexadesimal/ keword e.g.,dk`

  ```css
  h1{color:blue;font-size:15px;}
  /*
  	h1 = Selector
  
  	선언 = color:blue
  	color = property
  	blue = value
  	; = 구분자
  */
  ```

- Naver stylesheet

  https://pm.pstatic.net/css/main_v190709.css

  google > css beautify를 통해 내용을 복붙하면 깔끔하게 정리 가능

  회사에서는 일부러 uglify를 사용하여 비공개 처리

<br>

### CSS 활용하기

- Inline

  `h1 style="color:blue"`

- Embedding, 내부 참조

  HTML 내부에 CSS 포함

  ```html
  <head>
      <style>
          h1, p {
              color:blue
          }
      </style>
  </head>
  ```

- link file, 외부 참조

  외부의 CSS 파일 로드

  HTML에서의 `<style>` tag 내용을 CSS 파일로 분리하여 link를 걸어넣음

  ```HTML
  <!-- index.html -->
  <link rel="stylesheet" href="style.css">
  reslation : stylesheet
  href : file명
  
  <!-- style.css -->
  body{
    background-color: #f0f0f0
    background: #f0f0f0
  }
  h1, p{
    color: #3b3a30
  }
  ```
  
- 자주 사용되는 CSS properties

  https://developer.microsoft.com/en-us/microsoft-edge/platform/usage/

<br>

<br>

## Size 단위

- px

  device 별 pixel 크기는 제각각!

  대부분의 browser는 1px을 1/96 inch의 절대단위로 인식

- %

  %는 백분율 단위의 상대 단위

  요소에 지정된 사이즈(상속된 사이즈, 혹은 default 사이즈)에 상대적인 사이즈를 설정

- em

  배수 단위인 상대 단위

  요소에 지정된 사이즈(상속된 사이즈, 혹은 default 사이즈)에 상대적인 사이즈를 설정

  - em의 기준은 상속의 영향으로 유동적으로,

    상황에 따라 1.2em은 각기 다른 값을 소유하는 것이 가능

- rem

  최상위 요소(html)의 사이즈를 기준

  `rem`의 `r`은 root를 의미

- Viewport 단위

  device 마다 다른 크기의 화면을 보유하므로,

  상대적인 단위인 viewport를 기준으로 만든 단위

  IE 8 이하는 지원하지 않으며, 이후 버전의 IE와 Edge 또한 불완전한 지원

<br>

<br>

## Color 표현 방법

- HEX

  `#ffffff`

- RGB

  `rgb(0, 0, 0)`

- RGBA

  `rgb(0, 0, 0, 0.5)`

<br>

<br>

## Box Model 다루기

- 기본 속성 부여

  ```css
  div {
      /*상하좌우 10*/
      margin: 10px;
  }
  
  div {
      /*상하10 좌우20*/
      margin: 10px 20px;
  }
  div {
      /*상10 좌우20 하30*/
      margin: 10px 20px 30px;
  }
  div {
      /*상10 좌20 우30 하40*/
      margin: 10px 20px 30px 40px;
  }
  ```

- shorthand

  ```css
  div {
      margin: 10px;
      padding: 20px;
      border-width: 8px 4px 2px 1px;
      border-style: solid dotted dashed double;
      border-color: black blue pink gray;
  }
  ```

<br>

<br>

## class, 분류하기

- `class` property

  - `<div class="odd">`
  - `<div class="even">`

- unique property에 `id` 부여 및 `#`으로 선언

  ```css
  /*file.html*/
  <div id="zero">unique 공간</div>
  <div class="odd">공간1</div>
  <div class="even">공간2</div>
  <div class="odd">공간3</div>
  <div class="even">공간4</div>
  
  /*file.css*/
  #zero {
    background-color: Royalblue;
  }
  .odd{
    background-color: rosybrown;
  }
  .even{
  	background-color: green;
  }
  ```

<br>

<br>

## Visibility

1. visible

	해당 요소를 보이게함

2. hidden

	contents 구조를 망가뜨리지 않고, 안보이게 만들기 위해 사용

```css
 .hidden{
     visibility: hidden
 }
```

<br>

<br>

## Background-image

- `<body style="background-image: url(image 주소)">`

<br>

<br>

## Text & Font

- `font-size`, `font-family`, `letter-spacing`, `text-align`, `white-space`
- 상하 가운데 정렬 맞추기(좌우X)

- ```css
div {
    margin-left: auto;
    margin-right: auto;
    height: 100px;
    line-height: 100px;
    width: 300px;
    text-align: center;
  }
  ```

