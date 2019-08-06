# Bootstrap's Grid

---

- 가장 먼저는 가로(행) 배열을 고려

  그 이후 세로 정렬 형태 고려

- 현재는 대부분 표준화된 Grid(12개) 사용

  12개 Grid 분할은 2, 3, 4, 6개의 column으로 나누어 post를 띄움

  (약수가 많기 때문에 12가 표준으로 지정됨)

  https://www.w3schools.com/bootstrap4/bootstrap_grid_system.asp 참고

  https://poiemaweb.com/bootstrap-grid-system 참고

  i.g., `col-md-4` 의 경우 4개의 grid를 사용

  1. `<div class="container">`

     bootstrap을 활용한 grid system을 쓰려는 container를 생성

     - device 마다 px size가 다르며, 1100px size가 안정적으로 다양한 device를 커버하므로

       container 또한 1100px size로 고정

     - Naver의 main page 또한 이와 같으며, 심지어 반응형 페이지도 아니다.

     - `container-fluid`를 사용하면 wide screen에서도 전체 공간을 사용

  2. `<div class="row">`

     한 행 씩 정의

  3. `<div class="col">`

---

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
  
  - `col-12`를 추가 작성 시, mobile 버전에서 전체 grid를 사용
  
    ```html
    <div class="col-sm-4 col-12 card" style="width: 18rem;">
        <img src="https://images8.alphacoders.com/282/thumb-1920-282535.jpg" class="card-img-top" alt="spiderman">
        <div class="card-body">
            <h5 class="card-title">Ironman</h5>
            <p class="card-text">아이언맨을 못 보는게 아쉽다.</p>
            <a href="#" class="btn btn-primary">Visit Shield</a>
        </div>
    </div>
    ```