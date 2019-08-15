# Project 04 Review

---

## Navbar

- Navbar list 우측정렬

  1. div 에 flex 사용(개구리 옮기기 실습)

     ul 전체를 옮기므로, div

     ```html
     <div class="collapse navbar-collapse justify-content-end" id="navbarNav">
     ```

  2. ul 태그에 ml-auto

     ```html
     <ul class="navbar-nav ml-auto">
     ```

- Sticky Navbar

  - sticky-top

    ```html
    <nav class="navbar navbar-expand-lg navbar-light bg-light sticky-top">
    ```

- `bg-transparent`

  background 투명하게 만들기

  ```html
  <div class="collapse navbar-collapse justify-content-end bg-transparent" id="navbarNav">
  ```

---

## Header

- `<header>` 를 사용해도 무방

- `<section>`을 사용하여 `id="header"` 부여 가능

  `<div>`와 동일한 기능이지만, semantic하게 의미를 부여하기 위해 사용

  ```html
  <!-- Header 만들기 -->
  <section id="header" class="text-center d-flex align-items-center justify-content-center">
    <h2>당신에게 어울리는 영화를<br>추천해드립니다.</h2>
  </section>
  ```

  ```css
  #header {
    height: 350px;
    background-image: url("images/20183844-03.jpg");
    /* image size 늘림 */
    background-size: cover;
    /* 이미지 바둑판 배열 제거 */
    background-repeat: no-repeat;
    /* image와 color를 겹치게 사용 */
    background-color: gray;
    background-blend-mode: screen;
    /* image를 중앙으로 */
    background-position: 50%;
  }
  ```

---

## Footer

- Font Awesome으로 홈으로 가기 버튼 생성(arrow)

  ```html
  <!-- Footer 만들기 -->
  <!-- between을 이용하여 찢어놓기 -->
  <section id="footer" class="text-white px-3 d-flex justify-content-between align-items-center">
    <p class="mb-0">2019, Dongbin Cho</p>
    <!-- fa-2x 크기 두배 -->
    <a href="#header"><i id="top-btn" class="fas fa-arrow-circle-up fa-2x"></i></a>
  </section>
  ```

  ```css
  #footer {
    background-color: rgba(110, 120, 120, 1);
  }
  
  #top-btn {
    color: white;
  }
  ```

---

## Card

```html
 <section id="#movie-list">
    <div class="container">
      <h3 class="text-center my-5">영화 목록</h3>
      <hr id="title-underline">
      <div class="row">
        <!-- 반응형 제작 시 따로 div를 분리시키는것도 가능 -->
        <div class="col-lg-3 col-md-4 col-sm-6 col-12 my-3">
          <div class="card">
            <img src="images/20175771.jpg" class="card-img-top" alt="라이온 킹">
            <div class="card-body">
              <!-- span 태그와 badge class(button은 사이즈가 좀 크므로)를 이용하여 평점 넣기 -->
              <h4 class="card-title">라이온 킹 <span class="badge bg-info">9.03</span></h4>
              <hr>
              <p class="card-text">애니메이션<br>개봉일 : 2019.07.17.</p>
              <a href="#" class="btn btn-success">영화정보 보러가기</a>
            </div>
          </div>
        </div>
        <div class="col-lg-3 col-md-4 col-sm-6 col-12 my-3">
          <div class="card">
            <img src="images/20175771.jpg" class="card-img-top" alt="라이온 킹">
            <div class="card-body">
              <!-- span 태그와 badge class(button은 사이즈가 좀 크므로)를 이용하여 평점 넣기 -->
              <h4 class="card-title">라이온 킹 <span class="badge bg-info">9.03</span></h4>
              <hr>
              <p class="card-text">애니메이션<br>개봉일 : 2019.07.17.</p>
              <a href="#" class="btn btn-success">영화정보 보러가기</a>
            </div>
          </div>
        </div>
        <div class="col-lg-3 col-md-4 col-sm-6 col-12 my-3">
          <div class="card">
            <img src="images/20175771.jpg" class="card-img-top" alt="라이온 킹">
            <div class="card-body">
              <!-- span 태그와 badge class(button은 사이즈가 좀 크므로)를 이용하여 평점 넣기 -->
              <h4 class="card-title">라이온 킹 <span class="badge bg-info">9.03</span></h4>
              <hr>
              <p class="card-text">애니메이션<br>개봉일 : 2019.07.17.</p>
              <a href="#" class="btn btn-success">영화정보 보러가기</a>
            </div>
          </div>
        </div>
        <div class="col-lg-3 col-md-4 col-sm-6 col-12 my-3">
          <div class="card">
            <img src="images/20175771.jpg" class="card-img-top" alt="라이온 킹">
            <div class="card-body">
              <!-- span 태그와 badge class(button은 사이즈가 좀 크므로)를 이용하여 평점 넣기 -->
              <h4 class="card-title">라이온 킹 <span class="badge bg-info">9.03</span></h4>
              <hr>
              <p class="card-text">애니메이션<br>개봉일 : 2019.07.17.</p>
              <a href="#" class="btn btn-success">영화정보 보러가기</a>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
```

```css
#title-underline {
  background-color: orange;
  width: 70px;
}
```

---

## Modal & Courasel

- image에 Modal 적용

  ```html
  <div class="col-lg-3 col-md-4 col-sm-6 col-12 my-3">
            <div class="card">
              <!-- Modal 적용, 원래 modal은 html 하단에 모아서 기록 -->
              <img src="images/20175771.jpg" class="card-img-top" alt="라이온 킹" data-toggle="modal" data-target="#lionking">
  ```

- html 하단부의 Modal 및 Courasel

  ```html
  <!-- Live demo Modal -->
    <div class="modal fade" id="lionking" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel"
      aria-hidden="true">
      <div class="modal-dialog" role="document">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="lionking-title">라이온 킹, The Lion King</h5>
            <button type="button" class="close" data-dismiss="modal" aria-label="Close">
              <span aria-hidden="true">&times;</span>
            </button>
          </div>
          <div class="modal-body">
            <!-- Carousel 적용 -->
            <div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel">
              <ol class="carousel-indicators">
                <li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>
                <li data-target="#carouselExampleIndicators" data-slide-to="1"></li>
                <li data-target="#carouselExampleIndicators" data-slide-to="2"></li>
              </ol>
              <div class="carousel-inner">
                <div class="carousel-item active">
                  <img src="images/20175771-01.jpg" class="d-block w-100" alt="...">
                </div>
                <div class="carousel-item">
                  <img src="images/20175771-02.jpg" class="d-block w-100" alt="...">
                </div>
                <div class="carousel-item">
                  <img src="images/20175771-03.jpg" class="d-block w-100" alt="...">
                </div>
              </div>
              <a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">
                <span class="carousel-control-prev-icon" aria-hidden="true"></span>
                <span class="sr-only">Previous</span>
              </a>
              <a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">
                <span class="carousel-control-next-icon" aria-hidden="true"></span>
                <span class="sr-only">Next</span>
              </a>
            </div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
            <button type="button" class="btn btn-primary">Save changes</button>
          </div>
        </div>
      </div>
    </div>
  ```


---

## Tip

- body 색변하게 싸이킥하게

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

