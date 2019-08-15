# Bootstrap Class

---

## Grid

- `.container` : grid를 12개 가진 container를 생성

  `.row`

  `.col-[size]-[n_grids]`

  - size facctor

    `sm`

    `md`

    `lg`

    `xl`

  - n_girds factor

    1 ~ 12

### Device 종류에 의존하기

- `d-sm-none` : Mobile Screen

  `d-md-none` : Tablet Screen

  `d-lg-none` : 17inch 정도의 정사각형의 Screen

  `d-xl-none` : Wide Screen

---

## Position & Display

### Position

- `sticky-[where]`

  - where

    `top`

    `bottom`

- `p-[position]`

  - position factor

    `absolute`

    `relative`

    `fixed`

### Display

- `d-[display]`

  - display factor

    `inline`

    `inline-block`

    `block`

    `flex`

    `none`

- `float`- `left`/ `right` / `none`

---

## Aligning Tags

- `text-[where]` : `display`가 `inline` 인 text를 정렬

### Displlay-Flex 일 경우

- `justify-content-[where]`

  - where

    `center`

    `end`

    `between` : 요소들을 양 끝으로 붙인 후, 사이의 여백을 균등하게 정렬(양끝 여백 없음)

    `around` : 양 끝과의 간격까지 고려하여 모든 공간의 여백이 균등하게 정렬

- `align-items-[where]`

  - where

    justify-content와 동일

---

## Colors

- `bg-[color]`

  `text-[color]`

  - color

    `dark`

    `primary`

    `secondary`

    `light`

    `info`

    `warning`

    `danger`

    `success`

    `transparent` : 투명도 부여

---

## Margin & Padding

- `m-[value]`

  상하좌우 전체의 margin에  value를 부여

  - value(0 ~ 5)

    `0` : 0 rem (0 px)

    `1` : 0.25 rem (4px)

    `2` : 0.5 rem (8px)

    `3` : 1 rem (16px)

    `4` : 1.5 rem (24px)

    `5` : 3 rem (48px)

    `n1` : negative 1

    `n2` 

    `n3`

    `n4`

    `n5`

- `m[where]-[value]`

  where로 지정한 곳의 margin을 value만큼 조정

  - where

    `x` : 좌우

    `y` : 상하

    `l` : 좌

    `r` : 우

    `t` : 상

    `b` : 하

- `p-[value]`
- `p[where]-[value]`

---

## Text

- `font-weight-bold`
- `font-italic`