# SSAFY Week5 Day1

---

- 추천 개발 Tools

  - Trello

    동일 데이터, 고유값화를 어떻게 할 것인지 (감독명 동일, 어떻게 다르게 만들까)

    https://trello.com/?&aceid=&adposition=1t2&adgroup=54875417985&campaign=1018285860&creative=270057463393&device=c&keyword=trello&matchtype=e&network=g&placement=&ds_kids=p33209080176&ds_e=GOOGLE&ds_eid=700000001557344&ds_e1=GOOGLE&gclid=Cj0KCQjwj_XpBRCCARIsAItJiuRnaqBLpuuypLUkSMzzo3ApPe-nlUKHj6KxTkGsTCHuwP97kVLcj1UaAkWeEALw_wcB&gclsrc=aw.ds

    - 해야할 일, 완료한 일 등을 기록

      회사 게시판에 포스트잇 막 붙이는것과 유사한 기능

      GTD(Getting thins Done), To Do List의 협업 전용 버전 app

    - Kanban 방법론(간판)

      Slack을 Trello, Github을 연동시켜 사용 가능

      1. To Do
         - Contents 만들기
           - `Descrpition`
             - Markdown 사용 가능
           - `Checklist`
             - 공통 포트폴리오 페이지 5개 리서치
             - 공통 내용 분석
           - `Due Date`
           - `Labels`
             - label 별 이름 설정 및 task 구조화/시각화
         - html 구조 짜기
         - css 꾸미기
         - static page 배포

      2. Doing
         - To Do에서 옮겨와서 작업 시작
         - 하루 혹은 일주일 단위

      3. Done

    - `Create Team `

      - 이후 Edit Team Profile을 통해 `Short Name`을 변경하여 url 명을 변경 가능

        https://trello.com/choandkim

    ---

  - Notion

    Documentation Tool

    ---

  - Jira

  - `AWS route 53`

    Domain 관리

  - Go Daddy

  - 갓동주 추천도서 : 정리하는 뇌

  - momentum

    chorme extention, 깔끔한 배경사진, to do list

  - GIMP

    Open Source Photoshop
    
  - AWS re:Invent 2019
  
    개발자라면 죽기전에 가봐야 할 컨퍼런스
  
  - 창업 아이덴티티
  
    - 필수재로서의 아이덴티티, 감기약
    - 사치재로서의 아이덴티티, 비타민

---

## Intro to WEB Service

참고 자료 : 01_intro_to_web.pdf

- World Wide Web, WWW, W3

  인터넷에 연결된, 컴퓨터를 통해, 사람들이 정보를 공유할 수 있는 전 세계적인 정보 공간

  최초의 웹은 Next 회사의 OS Nexus에서 구동

  W3C(공기업)와 WHATWG(사기업)는 2019년 표준을 통합하는데에 합의

  - 요청, request
    - GET

      해당 Data를 달라고 요청

      url 뒤쪽에 params를 이용하여 보내기도 함

      HTML로 작성된 문서파일 하나를 받을 수 있다.

    - POST

      Data를 보내면서 처리해 줄 것을 요청
    
  - 응답, response

    요청 발생 시 Server에서 응답 발송

- Web Service 만들기

  = Server에서 요청과 응답을 처리할 프로그램을 개발한다

- Static Web

  단순한 웹 서비스로, 정적이며 정해진 응답만 가능

  요청이 단순할 것을 가정하며, 어떤 요청이든 동일한 답변만 가능

  e.g., Blogs, Portfolio page, Github Pages

- Web Browser

  클라이언트가 요청을 보내는 프로그램(수단)

  한국의 경우, 2016년 기준 Chrome이 IE를 역전

  - 왜 Chrome이 선호되고 있나

    Open Source의 세상인 현재, 표준화(W3C 지정)가 중요한 항목.

    과거 시장을 장악한 MS(IE)는 표준을 지키지 않는 활동을 많이 진행했고,

    현재 표준에 맞춰진 코드들이 잘 돌아가지 않는 상황이 도래.

  - Web을 사용한 접근

    - 내컴/dir_1/dir_2/filename.pdf

    - 남의컴/dir_1/dir_2/filename.pdf

      남의컴 주소 = 172.217.27.78

      - Domain 구매

        Domain Name System, DNS.

        전화번호부 처럼, Entity와 실제 주소를 매칭

        e.g., naver.com / google.com

        DNS Server가 중단되면 직접 주소를 입력해야 접속 가능

    - 사용자가 `Daum.net`을 주소창에 치면 발생하는 일

      1. Request URL을 DNS Server에 전송
      2. DNS Server는 원래 Daum의 주소를 브라우저에 전송
      3. Browser는 그 주소를 이용하여 Daum에 접속

    - 만약 Browser에서 F12, Network tab의 Remote Address를 통해 접속 시도 시,

      Proxing을 방지하기 위해 이러한 직접 접근을 Browser에서 차단

      - Proxy

        Client와 Server의 중간에서 정보를 빼내는 해킹 방법

---

## HTTP

참고 자료 : 02_html.pdf

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

- Web Developer Extension

  웹 개발자 필수 extention

  ---

- HTML 기본구조

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

      Beautify를 defaultFormatter로 설정하면 해당 파일을 이쁘게 구조를 변경

      설치 이후 `Ctrl + Shift + P > Indent Using Spaces`  2로 설정

      원하는 Code Block 선택 후 `Alt + Shift + F`

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

        e.g., 카톡으로 link를 보냈을 때 OG 내용을 표시

        e.g., Naver에서 `페이지 소스 보기`를 통해 og 내용과 image 확인 가능

    - body 요소

      Browser 화면에 나타나는 정보이며, 실제 내용

- Tag와 DOM TREE

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

  - 시맨틱 태그(Semantic Tag)

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

      e.g., Naver 검색 시 아름답게 부가 설명들을 게시

      - `spider`를 이용하여 자동으로 crawling

---

- Tips

  - `ol>li*4` + `Tab`

    `ol` tag 안에 `li` tag 4개를 추가하여 생성

  - 취소선

    `<strike>내용</strike>`

    `<s>내용</s>`

    `<del>내용</del>`

  - MDN Web Docs

    Web 개발에서의 Bible

    Firefox 개발사인 Mozila에서 제공하는 HTML Docs

  - `<a href="http://~~~">링크</a>` link 부여

  - `<table>` 조직화된 표 구조를 생성

    게시판 글들은 모두 `<table>`로 작성

