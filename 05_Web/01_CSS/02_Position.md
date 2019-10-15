# Position

1. Static, 기본 위치

   기본적인 요소의 배치 순서에 따라

   위에서 아래로, 왼쪽에서 오른쪽으로 순서에 따라 배치되며

   부모 요소 내에 자식 요소로서 존재할 때에는 **부모 요소의 위치를 기준**으로 배치

2. Relative, 상대 위치

   static 위치를 기준으로 위치를 변경하고 싶을 때 사용

   `top`, `right`, `bottom`, `left` 속성에 값을 부여하여 원하는 만큼 이동

3. Absolute, 절대 위치

   `<body>`의 `margin` 값을 고려하지 않으며, `<body>`를 벗어나서 위치를 고려

   - 부모  요소  또는  가장  가까이  있는  **조상  요소(static  제외, `<body>` 또한 static)를  기준**으로,

     좌표  프로퍼티(`top`,  `bottom`,  `left`,  `right`)만큼  이동

   - 즉,  **relative,  absolute,  fixed  프로퍼티가  선언되어  있는 부모  또는  조상  요소를  기준**으로  위치를  결정


   ```css
   .absolute {
     position: absolute;
     left: 190px;
     top: 100px;
   }
   ```

4. Fixed, 고정 위치

   - 부모  요소와  관계없이  **브라우저의  viewport를  기준**으로,

     좌표  프로퍼티(top,  bottom,  left,  right)을  사용하여  위치를  이동

   - **스크롤이  되더라도**  화면에서  사라지지  않고  항상  같은  곳에  위치

   - sticky navigation 등에 사용

   ```css
   .fixed {
     position: fixed;
     bottom: 0px;
     right: 0px;
     z-index: 2;
   }
   ```