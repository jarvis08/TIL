# Project 04

---

1. `01_layout.html`, `01_layout.css`

   `Navbar` Component, `<header>` & `<footer>` Tag,  `font` Attribute

   - `Navbar` Component

     - Navbar 상단 고정

       `<nav class="sticky-top">`

     - Home, Log In 과 같은 버튼 우측 정렬

       `<ul class="justify-content-end ml-auto">`

   - `<header>` 제작

     ```css
     header {
         width: 100%;
         height: 350px;
     }
     
     header > h1 {
         text-align: center;
         line-height: 350px;
         background-image: url("./images/20175771-01.jpg");
         background-repeat: no-repeat;
         background-position: center;
         color: lightgray;
     }
     ```

   - `<footer>`

     `fixed` position을 이용하여 footer 하단 고정

     - `font-family: "Times New Roman", Times, serif;` : footer 폰트 변경

     - `<p>` : `display: inline-block `
     - `<a>` : `float: right` home 이동 버튼 우측정렬

2. `02_movie.html`, `02_movie.css`

   `container` Class & `Card` Componenet

   - `<div class="container">` 를 이용하여 column을 나누고, `xl` `lg` `sm` 을 이용하여 반응형으로 제작

     Screen의 px size에 따라 아이템 표시 개수가 달라진다.

3. `03_detail_view.html`, `03_detail_view.css`

   `Modal` Component
   
   - modal 에 추가적인 image와 자세한 영화 설명을 기입