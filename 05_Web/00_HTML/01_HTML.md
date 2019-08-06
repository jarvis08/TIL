# HTML

참고 자료 : ./10_Lecture_Scripts/02_html.pdf

---

- HTTP(s)

  Hyper Text Transfer Protocol, 전송 규약

  s : secure

  - 보안에 있어서 HTTPs가 더 우수한데, 속도 또한 월등히 우수

    - 전체 data를 전송하는 http
    - hash 값을 전송하는 https

  - Hyper Text를 주고받는 규칙

    문서들을 선형적으로, 순서대로 보여주는 것이 아닌,

    Hyper Link를 통해 여러 page들을 순서 없이, 비선형적으로 오갈 수 있도록 하는 Text

- HTML

  Hyper Text Markup Language

  - data 별로 어떤 역할을 하는지 표시(markup)하는 문서 형태

    Web Page를 작성하기 위한 역할 표시 언어

  - Browser는 html 및 기타 markup language를 이쁘게 보여줌

  - MDN Web Docs

    Web 개발에서의 Bible

    Firefox 개발사인 Mozila에서 제공하는 HTML Docs

---

## HTML 기본구조

- Indentation은 2 space를 사용하는 것이 관례

  - VS Code 설정하기

    `Ctrl + Shift + P` > `Open Settings(JSON)`

    ```python
    # 추가하기
    "[html]": {
        "editor.tabSize": 2
    },
    "[css]": {
    	"editor.tabSize": 2
    }
    ```

  - VS Code Beautify Extention 설치

    Beautify를 defaultFormatter로 설정하면 코드를 올바른 구조로 손쉽게 교정 가능

    설치 이후 `Ctrl + Shift + P > Indent Using Spaces`  2로 설정

    단축키 : `Alt + Shift + F`

- DOCTYPE 선언부

  사용하는 문서의 종류를 선언

- HTML 요소

  문서의 root이며, head/body로 구분

  - head 요소

    Meta 정보(문서에 대한 문서)를 내포

    - 문서제목, 문자코드(인코딩)와 같은 문서 정보를 내포

    - Browser에 보여지지 않음

    - CSS 선언 혹은 외부 로딩 파일 지정 등을 작성

    - Open Graph

      어떤 정보를 가지고 있는지에 대한 요약본

      i.g., 카톡으로 link를 보냈을 때 OG 내용을 표시

      i.g., Naver에서 `페이지 소스 보기`를 통해 og 내용과 image 확인 가능

  - body 요소

    Browser 화면에 나타나는 정보이며, 실제 내용


---

## Tag와 DOM TREE

- DOM, Document Object Model

  javascript를 통해 객체를 다루며, tag들은 tree 형태를 가짐

  - `<html>` tag는 최상단

    `<html>` 아랫단 - `<head>`, `<body>`

    - `부모-자식`의 관계 유지

  - 이와 같은 tree 구조는 검색이 용이

- 주석, Comment

  `<!-- -->`

  Browser에 보여지지 않는 내용

- 요소, Element

  `<h1>`, `<iframe>`, `<form>` 등

  - HTML의 element는 tag와 contents로 구성

  - 대소문자를 구별하지 않으나, 소문자로 작성하는 것이 관례(Convention)

  - 요소간의 중첩 가능

  - `<img>` tag의 경우 contents가 필요 없을 것 같으나,

    엑박 혹은 시각 장애인들을 위해 `alt=""`(alternative 속성) 설명을 작성하는 것이 관례

- 속성, Attribute

  `<a href="https://google.com"/>`

  `href` : 속성명

  `google.com` : 속성 값

  tag에는 attribute가 지정될 수 있다.

  - 띄어쓰기 없이 사용하며, `""`를 사용하는 것이 관례

  - `id`, `class`, `style` 속성은 tag와 상관 없이 모두 사용 가능

- DOM tree

  tag는 중첩 사용이 가능하며,

  중첩 사용 시 중첩 이전의 부모 관계와 동일한 관계를 갖음

  ```html
  <body>
      <ul>
          <!-- li tag들은 모두 같은 부모를 소유 -->
          <!-- li tag들은 모두 형제(sibling) 관계 -->
          <li></li>
          <li></li>
          <li></li>
      </ul>
  </body>
  ```

---

## 시맨틱 태그, Semantic Tag

- 공간 분할에 사용되어 온 Division Tag, `<div></div>`, `<span>`
  - 공간을 분할할 뿐 의미는 없다.
  - display: block을 지정하기 위한 기본 레이아웃 태그

- 의미 없는 분할(`div`)을 개선하고자 Semantic Tag를 제시

  Google News, Naver News 등의 사이트에서 `페이지 소스 보기`를 통해 확인 가능

  - `header`

    헤더(문서 전체나 섹션의 헤더)

  - `nav`

    내비게이션

  - `aside`

    사이드에 위치한 공간으로, 메인 콘텐츠와 관련성이 적은 콘텐츠에 사용

  - `section`

    문서의 일반적인 구분으로 컨텐츠의 그룹을 표현하며, 일반적으로 h1~h6 요소를 가짐

  - `article`

    문서, 페이지, 사이트 안에서 독립적으로 구분되는 영역(포럼/신문 등의 글 또는 기사)

  - `footer`

    푸터(문서 전체나 섹션의 푸터)

- 개발자 및 사용자 뿐만 아니라 검색엔진(구글, 네이버) 등에

  의미 있는 정보의 그룹을 태그로 표현하여

  단순히 보여주기 위한 것을 넘어서 의미를 가지는 태그들을 활용하기 위한 노력!

- Sementic Tagging의 좋은 예시

  `Google News > Web Developer Extention > Information > View Document Outline`

  **검색 엔진 최적화(SEO, Search Engine Optimization)**가 아주 중요한 개념으로 떠올랐으며, 이를 위해 Semantic하게 정리하여 **page building**부터 신경쓰는 것이 중요.

- Google 검색 시, SEO가 잘 되어진 web site는 검색창에 검색 만으로도 Link 밑에 Site에 대한 설명들이 보여짐

  i.g., Naver 검색 시 아름답게 부가 설명들을 게시

  - `spider`를 이용하여 자동으로 crawling

