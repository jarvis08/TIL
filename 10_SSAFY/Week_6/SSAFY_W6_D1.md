# SSAFY Week6 Day1

---

- 복합 선택자

  선택자들을 여러개 섞어 사용하기

  `>`, `~`, ` `, `*`, `+`
  
    - 자식 셀렉터와 후손 셀렉터

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

  - 형제 셀렉터
  
    ```css
    /* + 는 형제를 의미 */
    /* a + ul은 a와 바로 붙어있는 ul에만 적용 */
    a + ul {
      background-color: gold;  
    }
    
    /* ~ 은 형제를 의미하지만, 형제관계인 모든 ul에 적용 */
    a ~ ul {
      border: solid darkgray 1px;
    }
    ```
  
- 속성 셀렉터

  e.g., `<img src="사진.jpg" target="_blank">` 태그 안의 `src="주소"` 내용 혹은 `target=_blank`

  - `target=_black` : 새 탭에서 링크 열기
  - `<a>` 태그 중 `target`이 `_black` 인 것들에 대해 속성 부여

      ```css
      /* 속성 셀렉터 */
      a[target="_blank"] {
        border: solid 2px black;
        border-radius: 10px;
        padding: 5px;
      }
      ```

- class 셀렉터

  ```css
  /* me class 셀렉트 */
  .me {
    background-color: black;
    color: white;
  }
  ```

- Regular Expression, 정규표현식 활용

  정규표현식 참고 사이트, regexr.com

  ```css
  /* Regular Expression, 정규 표현식의 개념
  class를 주로 활용하며, 정규 포현식은 있긴 있는데 잘 안쓰는 방법 */
  /* img 태그의 alt 속성에 TYPE이라는 값이 포함되어 있다면 */
  img[alt|="TYPE"] {
  
  }
  /* $ ^ ~ 등의 기호도 존재 */
  ```

---

## Bootstrap

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

  - 개별 end-user의 가까운 서버를 통해 빠르게 전달

  - 외부 서버를 활용함으로써 본인 서버의 부하 감소

  - 보통 적절한 수준의 캐시 설정으로 빠르게 로딩 가능

    캐시 : 사용한 경험이 있는 것은 저장하여, 더 빠르게 로딩

  1. Utilities

     편하게 클래스로 적용

     1. Spacing

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

        - `m`, `mt`, `mb`, `ml`, `mr`, `mx`, `my`, `m-n1~4 `, `mx-autio`

          = 전체, top, bottom, left, right, x축, y축, `-(negative)`

        - mx-auto, ml-auto, mr-auto의 경우 잘 쓰이지 않는다.

          주로  `position: flex`와 `grid`를 이용하여 설정

     2. Color

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

          - `class="navbar-dark .bg-primary"`

     3. Border

        - `class="border border-success rounded"`

        ```html
          <div class="border border-success rounded">
            <p>보더 테스트</p>
          </div>
        ```

     4. Display

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

     5. Position

        `class="position-static/relative/absolute/fixed/float"`

        - `class="fixed-top/bottom"`

     6. Text

        https://getbootstrap.com/docs/4.3/utilities/borders/

        공식 홈페이지 Unitilities에서 자세하게 확인 가능

        - `class="text-center"`

          `class="font-weight-bold"`

          `class="font-italic"`

- **color와 breakpoint는 공식 홈페이지의 Components에서 자세히 확인 가능**

  https://getbootstrap.com/docs/4.3/components/alerts/

  - 자주 쓸 Components

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

---

## Grid

- 가장 먼저는 가로(행) 배열을 고려

  그 이후 세로 정렬 형태 고려

- 현재는 대부분 표준화된 Grid(12개) 사용

  12개 Grid 분할은 2, 3, 4, 6개의 column으로 나누어 post를 띄움

  (약수가 많기 때문에 12가 표준으로 지정됨)

  https://www.w3schools.com/bootstrap4/bootstrap_grid_system.asp 참고

  https://poiemaweb.com/bootstrap-grid-system 참고

  e.g., `col-md-4` 의 경우 4개의 grid를 사용

  1. `<div class="container">`

     bootstrap을 활용한 grid system을 쓰려는 container를 생성

     - device 마다 px size가 다르며, 1100px size가 안정적으로 다양한 device를 커버하므로

       container 또한 1100px size로 고정

     - Naver의 main page 또한 이와 같으며, 심지어 반응형 페이지도 아니다.

     - `container-fluid`를 사용하면 wide screen에서도 전체 공간을 사용

  2. `<div class="row">`

     한 행 씩 정의

  3. `<div class="col">`

     - 12개 까지는 자동으로 등분하여 grid 점유

         ```html
         <div class="container-fluid bg-info">
             <div class="row">
                 <div class="col">
                     칼럼
                 </div>
                 <div class="col">
                     칼럼
                 </div>
                 <div class="col">
                     칼럼
                 </div>
             </div>
         </div>
         ```

     - grid 점유 개수 지시 가능

         ```html
         <div class="container-fluid bg-info">
             <div class="row">
                 <div class="col-3 bg-light">
                     칼럼
                 </div>
                 <div class="col-6">
                     칼럼
                 </div>
                 <div class="col-3 bg-light">
                     칼럼
                 </div>
             </div>
         </div>
         ```

     - gird 별 간격을 부여하고 싶을 때는 padding을 사용
  
       만약 이상태로 margin을 주면 포화로 인해 삐져나가며 어긋난 형태를 보임
     
       padding 부여 시, content 영역을 줄이고 간격 역할을 하는 padding을 추가
     
       ```html
       <div class="container-fluid bg-info">
           <div class="row">
               <div class="col-3 px-1 bg-light">
                   칼럼
               </div>
               <div class="col-6 px-1">
                   칼럼
               </div>
               <div class="col-3 px-1 bg-primary">
                   칼럼
               </div>
           </div>
       </div>
       ```
     
     - 반응형의 경우 mobile 기준으로 생각
     
       small device일 경우 grid를 해제하여 한 row에 한 content를 담도록 변경
     
       `sm`, `md`, `lg`, `xl`..
     
       ```html
       <div class="container-fluid bg-info">
           <div class="row">
               <!-- 반응형 device breakpoint: sm 4 / x축 패딩 1 / 배경색 light -->
               <div class="col-sm-4 px-1 bg-light">
                   칼럼
               </div>
               <div class="col-sm-4 px-1">
                   칼럼
               </div>
               <div class="col-sm-4 px-1 bg-primary">
                   칼럼
               </div>
           </div>
       </div>
       ```
     
       - 단계적으로 줄이기
     
         `lg`(large device)의 경우 990px 즈음 `md`(medium device) 형태로 변경
     
         요소 검사 시 우측 상단에 px 표시되어 확인 가능
     
         ```html
         <div class="container-fluid bg-info">
             <div class="row">
                 <!-- large device일 때에는 3칼럼, medium device일 때에는 6칼럼을 차지 -->
                 <div class="col-lg-3 col-md-6 px-1 bg-light">
                     칼럼
                 </div>
                 <div class="col-lg-3 col-md-6 px-1 bg-primary">
                     칼럼
                 </div>
                 <div class="col-lg-3 col-md-6 px-1 bg-light">
                     칼럼
                 </div>
                 <div class="col-lg-3 col-md-6 px-1 bg-primary">
                     칼럼
                 </div>
             </div>
         </div>
         ```
     
         