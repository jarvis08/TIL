# 선택자, Selector

## 복합 선택자

- 선택자들을 여러개 섞어 사용하기

  `>`, `~`, ` `, `*`, `+`

<br>

<br>

## 자식 셀렉터 & 후손 셀렉터

- html

  ```html
  <div>
  	<h1>
  		이건 테스트입니다.
  		<span>이건 inline 태그인 span</span>
    </h1>
  </div>
  ```

- css

  ```css
  /* 자식 셀렉터, 후손 셀렉터 */
  /* 자식 셀렉터 :: div와 span이 직접 연결이 되지 않아 적용 X */
  div > span {
    color: burlywood
  }
  
  /* 후손 셀렉터 :: 직접 연결되어 있지 않지만, div의 후손으로 span이 있기만 한다면 다음을 적용 */
  div span {
    color: burlywood
  }
  ```

- Exercise. 자식/후손 색칠하기

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

<br>

<br>

## 형제 셀렉터

- `+`

  `a + ul`은 `a`와 바로 붙어있는 `ul`에만 적용

  ```css
  a + ul {
    background-color: gold;  
  }
  ```

- `~`

  형제 관계인 모든 `ul`에 적용

  ```css
  a ~ ul {
    border: solid darkgray 1px;
  }
  ```

<br>

<br>

## 속성 셀렉터

e.g., `<img src="사진.jpg" target="_blank">` 태그 안의 `src="주소"` 내용 혹은 `target=_blank`

- `target=_black` : 새 탭에서 링크 열기

- `<a>` 태그 중 target이 _blank 인 것들에 대해 속성 부여

  ```css
  a[target="_blank"] {
    border: solid 2px black;
    border-radius: 10px;
    padding: 5px;
  }
  ```

<br>

<br>

## class 셀렉터

```css
/* me class 셀렉트 */
.me {
  background-color: black;
  color: white;
}
```

<br>

<br>

## Regular Expression, 정규표현식 활용

- 정규표현식 참고 사이트, regexr.com

```css
/* Regular Expression, 정규 표현식의 개념
class를 주로 활용하며, 정규 포현식은 있긴 있는데 잘 안쓰는 방법 */
/* img 태그의 alt 속성에 TYPE이라는 값이 포함되어 있다면 */
img[alt|="TYPE"] {

}
/* $ ^ ~ 등의 기호도 존재 */
```