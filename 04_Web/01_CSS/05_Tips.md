# Tips for CSS & Bootstrap

## Animate.css

- animation 효과 부여

  https://daneden.github.io/animate.css/

  ```html
  <!-- CDN -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/3.7.2/animate.min.css">
  ```

<br>

<br>

## Font Awesome

- github, facebook 같은 icon 배포

  https://fontawesome.com/

  ```html
  <!-- CDN -->
  <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.10.1/css/all.css" integrity="sha384-wxqG4glGB3nlqX0bi23nmgwCSjWIW13BdLUEYC4VIMehfbcro/ATkyDsF/AbIOVe" crossorigin="anonymous">
  ```

- size 조절

  https://fontawesome.com/how-to-use/on-the-web/styling/sizing-icons

  ```html
  <i class="fas fa-camera fa-xs"></i>
  <i class="fas fa-camera fa-sm"></i>
  <i class="fas fa-camera fa-lg"></i>
  <i class="fas fa-camera fa-2x"></i>
  <i class="fas fa-camera fa-3x"></i>
  <i class="fas fa-camera fa-5x"></i>
  <i class="fas fa-camera fa-7x"></i>
  <i class="fas fa-camera fa-10x"></i>
  ```

<br>

<br>

## Codepen

- God Designers' open source

  https://codepen.io/popular/pens/

<br>

<br>

## Etc

- color
  - black : `#f0f0f0`
    - light-black: `#444444`
- white : `#030303`
  - link-blue: `rgb(15, 168, 224)`
- freecodecamp에서 bootstrap무료 수강 및 수료증

<br>

<br>

## body tag 색상변하게하기

```css
body {
  margin: 0;
  width: 100%;
  height: 100vh;
  color: black;
  background: linear-gradient(-45deg, #fca084, #fc74a8, #4683f5, #6cffdd);
  background-size: 500% 700%;
animation: gradientBG 7s ease infinite;
}
@keyframes gradientBG {
  0% {
      background-position: 0% 50%;
  }
  50% {
      background-position: 100% 50%;
  }
  100% {
      background-position: 0% 50%;
  }
}
```