# SSAFY Week5 Day2

참고 자료 : 03_css.pdf

---

## CSS

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

- CSS 활용하기

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
    }
    h1, p{
      color: #3b3a30
    }
    ```

- 자주 사용되는 CSS properties

  https://developer.microsoft.com/en-us/microsoft-edge/platform/usage/

- size

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

- color

  - HEX

    `#ffffff`

  - RGB

    `rgb(0, 0, 0)`

  - RGBA

    `rgb(0, 0, 0, 0.5)`

  ---

- **Box Model**

  모든 HTML은 네모를 사용하여 만듬

  원은 네모를 돌려깎아 사용

  `Margin` > `Border` > `Padding` > `Content`

  1. Margine

     테두리 바깥의 외부 여백

     배경색 지정 가능

     `<p>` `<h1~>` 는 기본적으로 margine이 있어, 줄 띄우기가 가능

  2. Border

     - 테두리 영역

       `padding`과 `margin`의 경계를 테두리로 설정

       `solid` 등의 모양 설정 가능

       - MDN Docs

         https://developer.mozilla.org/ko/docs/Web/CSS/border-style

  3. Padding

     테두리(border) 안쪽의 내부 여백

     요소에 적용된 배경의 컬러/이미지는 패딩까지 적용

  4. Content

     실제 내용이 위치

- Box Model 다루기

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

  ---

- 분류하기

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

  ---

- **Display**

  1. block

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

  2. inline

     - 문장 삽입 가능

     - content의 너비만큼 가로폭을 차지

       따라서 font size가 커지면 inline의 width 증가

     - `height`, `width`, `margin-top/bottom` 등의 property 지정 불가!

       - `margin,padding-left/right` property 지정 가능
       - 하지만 `auto` 지정은 불가

     - 상/하 여백은 `line-height`로 지정

     - Inline level element

       - `<a>`, `<span>`, `<img>`, `<br>`, `<strong>`

       - `<form>` 내부에서 사용되는 tag들

          `<input>`, `<select>`, `<textarea>`, `<button>`

  3. Inline-Block

     inline level element 처럼 한 줄에 표시되지만,

     block에서의 width, height, margin(top, bottom) property를 모두 지정 가능

     ```css
     <div class="inline-block">inline-block<span>붙여쓰기</span></div>
     
     .inline-block {
       display: inline-block;
       margin: 50px;
       height: 100px;
       width: 100px;
     }
     ```

  4. None

     해당 요소를 화면에 표시하지 않으며, 공간 조차 사라지게 조치

     동적으로 작업할 때 사용

     공간은 유지하고 싶을 때, `visibility`를 사용

     *e.g., log-in 시 log-in 버튼 사라지게 하기*

     ```css
     <div class="none">얘는 곧 사라집니다.</div>
     
     .none {
         display: None;
     }
     ```

  ---

  - Visibility 속성

    1. visible

       해당 요소를 보이게함

    2. hidden

       contents 구조를 망가뜨리지 않고, 안보이게 만들기 위해 사용

       ```css
       .hidden{
           visibility: hidden
       }
       ```

  ---

  - Background-image

    `<body style="background-image: url(image 주소)">`

  - Text & Font

    `font-size`, `font-family`, `letter-spacing`, `text-align`, `white-space`

    - 상하 가운데 정렬 맞추기(좌우X)

      ```css
      div {
        margin-left: auto;
        margin-right: auto;
        height: 100px;
        line-height: 100px;
        width: 300px;
        text-align: center;
      }
      ```

  ---

  - Position

    1. Static, 기본 위치

       기본적인 요소의 배치 순서에 따라

       위에서 아래로, 왼쪽에서 오른쪽으로 순서에 따라 배치되며

       부모 요소 내에 자식 요소로서 존재할 때에는 부모 요소의 위치를 기준으로 배치

    2. Relative, 상대 위치

    3. Absolute, 절대 위치

       `<body>`의 `margin` 값을 고려하지 않으며, `<body>`를 벗어나서 위치를 고려

       - 부모  요소  또는  가장  가까이  있는  조상  요소(static  제외, `<body>` 또한 static)를  기준으로,

         좌표  프로퍼티(`top`,  `bottom`,  `left`,  `right`)만큼  이동

       - 즉,  relative,  absolute,  fixed  프로퍼티가  선언되어  있는

         부모  또는  조상  요소를  기준으로  위치를  결정

       ```css
       .absolute {
         position: absolute;
         left: 190px;
         top: 100px;
       }
       ```

    4. Fixed, 고정 위치

       - 부모  요소와  관계없이  브라우저의  viewport를  기준으로,

       - 좌표  프로퍼티(top,  bottom,  left,  right)을  사용하여  위치를  이동

       - **스크롤이  되더라도**  화면에서  사라지지  않고  항상  같은  곳에  위치

         sticky navigation 등에 사용

       ```css
       .fixed {
         position: fixed;
         bottom: 0px;
         right: 0px;
         z-index: 2;
       }
       ```

  ---

- grid

  격자 형식으로, Web Page의 구간을 나누어 규격을 맞추는 방법

---

## HTML

- `<blockquote cite="http://"></blockquote>` : 인용문 태그

- id/pw 입력 창

  ```html
  <form action="">
      <span>ID : </span><input type="text" palceholder="user"><br>
      <span>PWD : </span><input type="password" palceholder="****"><br>
      <button type="submit">로그인</button>
  </form>
  ```
  
- selector, 자식 색칠하기

  ```html
  <head>
    <style>
      #ssafy > p:nth-of-type(2) {
        color: red
      }
    </style>
  </head>
  
  <body>
    <div id="ssafy">
      <h2>어떻게 선택될까/</h2>
      <p>첫번째 달락</p>
      <p>두번째 달락</p>
      <p>세번째 달락</p>
      <p>네번째 달락</p>
    </div> 
  </body>
  <!-- 두번째 달락이 빨간색으로 색칠됨 -->
  ```

  ```html
  <head>
    <style>
      #ssafy > p:nth-child(2) {
        color: red
      }
    </style>
  </head>
  
  <body>
    <div id="ssafy">
      <h2>어떻게 선택될까/</h2>
      <p>첫번째 달락</p>
      <p>두번째 달락</p>
      <p>세번째 달락</p>
      <p>네번째 달락</p>
    </div> 
  </body>
  <!-- 두번째 달락이 빨간색으로 색칠됨 -->
  ```

  - 후손 선택자와 자식 선택자 차이

    후손 셀렉터는 자식 셀렉터와 달리 모든 후손(대대손손)에 적용

---

- 절대 경로

  - Windows
    - root 경로 = `C:\` 
  - Linux
    - root 경로 = `/`
  - `~/` : home directory

- 상대 경로

  root 경로 = 자기 자신(working directory)

  - `../../directory/`

    두 단계 거슬러 올라간 후, direc 이라는 directory

---

- color
  - black : `#f0f0f0`
    - light-black: `#444444`
- white : `#030303`
  - link-blue: `rgb(15, 168, 224)`
  
- freecodecamp에서 bootstrap무료 수강 및 수료증

---

## bootsrap

- card 여러개 한 줄로 집어넣기

```html
<div class="container">
    <div class="row">
		카드들의 코드
    </div>
</div>
```

- lorem pixel

  random image 부여

- start bootstrap

  무료 bootstrap theme

---

## virtual env

```shell
mkdir python-verualenv
# -m : modul
python -m venv ~/python-vertualenv/3.7.3
cd work-directory
# 가상환경 실행
source ~/python-virtualenv/3.7.3/Scripts/activate
# 가상환경 종료
deactivate

# alias 이용하여 실행 코드 줄이기
# .bashrc 추가
alias venv='source ~/python-virtualenv/3.7.3/Scripts/activate'
source .bashrc

```



