# Javascript Intro

Javascript는 Adobe의 Action Script를 대체한다.

우리는 Web Browser를 이용하여 html 페이지를 꾸며진 형태로 볼 수 있으며,

Javascript를 사용하여 Web Browser를 조작할 수 있다.

Crawling은 HTML 문서에 포함된 데이터를 가져오는 것이므로, Javascript로 데이터를 가져온 페이지의 경우 크롤링이 불가하다. 따라서 셀레니움과 같은 도구를 사용하여, 렌더 되어있는 형태의 데이터를 가져와야 한다.

`utm_source`같은 사용자 접속 경로 tracking 하는 태그들 또한 Javascript로 구현한다.

<br>

### JS 배경

- Browser Wars
  - Macintosh: Netscape, 최초의 브라우저
  - Macintosh: Safari
  - Microsoft: Internet Explorer, Edge
  - Microsoft: Edge
  - Google: Chrome
  - Mozilla
  - Camino
  - Firefox
  - Opera

JavaScript는 Java의 인기에 편승하기 위해 작명되었을 뿐, Java와 전혀 무관한 언어입니다.

과거에는 IE.JS, Chrome.JS, FF.JS 등 **브라우저 조작 언어**였으며, 브라우저 별로 버전이 존재했습니다. 이후 ECMA International은 기술 규격을 통합하기 위해 **ECMA Script** 제작했으며, 현재의 JavaScript는 정확히 말하자면 ECMA Script를 말합니다. Chrome은 2015년 부터 **ES2015(ES6)**를 따를 것을 발표하며 본격적으로 활용되기 시작했습니다. 현재 판매되고 있는 JavaScript 서적들만 해도 ES6를 따르는 내용임을 쉽게 찾아볼 수 있습니다.

*ECMA International:*
:*주로 H/W 규격들을 조정하여 호환성을 높이는 연구를 하는 기관으로, C# Json CD-ROM CLI C++ 등의 규격 제작*

<br>

### JavaScript Frameworks

pylint처럼 eslint 또한 존재한다.

- **Vanila JS**: 프레임워크를 사용하지 않은 JS

- **jQuery**

  - John Resig이 개발했으며, JS를 정말 간편하게 사용할 수 있도록 도와준다.
  - 존 레식은 TIL의 시초인 사람으로도 유명

- 멸종되고 있는 자바스크립트의 프레임워크

  - Ember.js
  - Backbone.js

- 현재 많이 사용되는 프레임워크

  - **React.js**

    가장 많이 사용되는 프레임워크

    - React Native, RN

      외주의 왕, 코드 복붙하여 새로운 어플리케이션 생성 가능

      아무리 고단수여도 1개월 이상 걸리는 프로젝트를, 2주 안에 가능

  - Angular.js

  - **Vue.js**

<br>

### Window

Window의 구성은 다음과 같습니다.

- **DOM, Document Object Model**

  - `document` object

    html 페이지의 속성을 저장

    i.g., `window.document.title`: Browser 상의 맨 위 tag에 표시되는 제목을 저장

- **BOM, Browser Object Model**

  - `navigator` object
  - `screen` object
  - `location` object
  - `frames` object
  - `history` object
  - `XMLHttpRequest`: 특정 End Point로 브라우저의 Request를 보내주는 역할

- **JavaScript**

  - `Object`
  - `Array`
  - `Function`

예시: `windws.innerWidth`라는 객체는 브라우저 크기에 따라 속성 값이 달라집니다.

<br>

### Object

OOP: 세상의 사물을 그대로 옮겨오기 위해 제작한 개념

- Property:사물의 고유한 값(속성)을 가질 수 있다.
- Method: 사물이 행하는 동작이 존재한다.

객체는 꼭 클래스가 필요한 것은 아니며, JS의 경우 대표적인 그 예시입니다. **클래스 선언 없이 객체를 생성하고 조작**합니다.

처음에는 클래스라는 개념이 없었으나, 이후 필요에 의해 사용합니다.

JS에서 Object는 **key-value 구조**를 가집니다.

- JSON과의 차이점

  ```
  Object는 내부에 함수를 선언하여 호출할 수 있다.
  key를 생성할 때 string일 지언정 '' 혹은 ""를 사용하지 않아도 된다.
  i.g., JSON은 'name'이라고 key를 설정해야 하지만, object는 name 이라고만 설정해도 무방하다.
  ```

<br>

### Node.js

[Node.JS](https://nodejs.org/ko/)

JavaScript는 브라우저 조작 언어였으며, 브라우저 상에서만 작동이 가능했습니다. 하지만 Node.js가 **JavaScript Runtime**을 제작하며, fully functioning programming language로 사용 가능해졌습니다. Node.js는 Google이 Chrome에서 JS를 사용하던 중 제작했으며, 현재는 Linux Foundation이 관리하고 있습니다.

```bash
$ node -v
v12.13.0

$ npm -v
6.12.0
```

<br>

<br>

## JS 문법

Semicolon(`;`)은 더이상 붙이지 않는 [추세](https://twitter.com/BrendanEich/status/951554266535141377)입니다. JS의 창조자인 브랜든 아이크(Brendan Eich)가 붙이지 않는게 좋다고 개인 트위터를 통해 의견을 제시했습니다.

- `//`: 주석

- `let x = 3`

  - 지역 변수를 선언할 때 사용하며, 동일 이름의 지역 변수는 선언이 불가하다.
  - `x = 1` 전역 변수로 선언한 후에 지역변수로 `let x = 3`을 선언하면, 전역 변수를 덮어쓸 수 없으므로 에러가 발생한다.

  ```javascript
  // Block Scope 확인하기
  x = 2
  if (x == 2) {
      let x = 3
      console.log(x)
  }
  console.log(x)
  // result
  // 3
  // 2
  ```

  Python의 경우 `if`, `for`문은 함수와 달리 block scope로 취급되지 않기 때문에 `3 3`이 출력될 것입니다.

  ```javascript
  x = 2
  if (x == 2) {
      let x = 3
      let y = 4
      console.log(x)
  }
  console.log(x)
  console.log(y)
  // result
  // ReferenceError: y is not defined
  ```

- `const 대문자 = 값`: 상수 선언

  - `const`는 Declare(선언) 시 추가적인 Assign(할당) 조차 불가합니다.

- Single quotation marks(`''`)를 사용하여 **Concatenation**을 할 수 있으며,

  Backtick(`)을 사용하여 **Interpolation**을 할 수 있습니다.

  ```javascript
  const MY_FAV = 13
  // concatenation
  console.log('내가 좋아하는 숫자는 ' + MY_FAV)
  // interpolation
  console.log(`내가 좋아하는 숫자는 ${MY_FAV}`)
  ```

- `var`: 더 이상 사용되지 않는 문법입니다.

<br>

### Declarative Programming

- **명령형 프로그래밍, Imperative Programming**

  모든걸 개발자가 명령합니다.

- **선언적 프로그래밍, Declarative Programming**

  큰 그림만 그려주면, React / Vue와 같은 Framework들이 알아서 자세한 내용을 처리해 줍니다. Django의 ORM도 선언적 프로그래밍을 구사하게 해줍니다.