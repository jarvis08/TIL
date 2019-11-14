# Template Inheritance

- DRY, Do not Repeat Yourself

- Navbar라는 요소는 모든 페이지에 있어야 하므로 다른 페이지에서도 가능해야 한다.

  따라서 이를 복사&붙여넣기 보다는 상속을 이용하여 진행

  1. 공통적으로 사용할 템플릿(코드)을 추출

  2. 해당 템플릿(코드)를 파일로 따로 생성

     `FIRST_APP/first_app/templates/base.html`

     - **body 끝부분에 block을 설정**

       코드 구멍을 뚫는 역할이며, **상속받는 페이지의 내용들이 들어갈 곳**을 정의

     ```html
     <!-- base.html -->
     <!-- inherite 할 html 내용들 -->
     <!-- body는 내가 설정하는 이름이며, 관례로 body 혹은 content -->
       {% block body %}
       {% endblock %}
     </body>
     ```

  3. 활용할 다른 템플릿 파일에서 불러와서 사용

     **상속할 template(base.html)과 중복되는 내용은 모두 삭제**

     - `base.html`을 상속받는다는 코드를 가장 위에 작성

       ```html
       {% extends 'base.html' %}
       ```

     - `base.html`에서 설정한 block에 들어갈 내용들(상속 받지 않는)을 작성

       ```html
       {% block body %}
       
         <h2>For문</h2>
         <p>{{ myname }}</p>
         <p>{{ class }}</p>
         {% for item in class %}
           <p>{{ item }}</p>
         {% endfor %}
       
       {% endblock %}
       ```

     - `home.html` 전체 코드

       ```html
       <!-- home.html -->
       <!-- inherite 받는 html -->
       {% extends 'base.html' %}
       {% block body %}
       
         <h2>For문</h2>
         <p>{{ myname }}</p>
         <p>{{ class }}</p>
         {% for item in class %}
           <p>{{ item }}</p>
         {% endfor %}
       
       {% endblock %}
       ```