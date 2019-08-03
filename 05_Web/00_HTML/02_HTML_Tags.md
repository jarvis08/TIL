# HyperText Markup Language

------

## Tags

- `<!DOCTYPE html>` : html5 규칙으로서, html임을 명시

  `<html>` : 시작과 끝에 부여

  `<!-- -->` : 주석

  `&nbsp` : 공백 2칸

- `<head>`

  - `<title>title</title>` : browser tab title

- `<body>` : browser page 내용

  `<h1~6>` : 대제목 ~ 소제목

  `<br>` : 줄 띄우기(사용X, 망가뜨림)

  `<ol>` : ordered line, 번호 숫자 부여

  `<ul>` : unordered line, 쩜

  `<dl>` : definition line

  `<li>` : ol, ul, dl의 각 항목 line

  `<p>` : paragraph, 본문 작성

  `<a href="">words</a>` : link

  - 취소선
    - `<strike>내용</strike>`
    - `<s>내용</s>`
    - `<del>내용</del>`

  `<img width= height= src="">` : image

  `<iframe width= height= src="">` : video

  `<table>` : 조직화된 표 구조를 생성, 모든 게시판에 사용됨

---

## Emmet

- `ol>li*3` + `Tab` 

  make 3 `<li>` tags in a `<ol>` tag

---

## Extensions

- VS Code extention - `Live Server`

  VS Code 좌측 > Explorer 탭 > html 파일 우클릭 > Open with Live Server

- `Web Developer`

  Web Page를 벗겨보고, 들여다볼 수 있는 도구

  i.g., CSS 없애기

---

## Ex

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