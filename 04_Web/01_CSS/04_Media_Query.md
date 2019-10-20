# Media Query

### viewport

- webpage에서 보이는화면의 크기이자 기준 폭

- apple에서 처음 개발

- pc의 경우 browser 크기
- `content`
  - `width`와 `initial-scale`을 정의
  - `device-width` 설정을 위해 만들어진 속성
  - `initial-scale=1.0` : 1배율로 시작함을 의미

```html
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="ie=edge">
  <title>Document</title>
</head>
```

<br>

### media query

`@media (조건) { 내용 }`

viewport를 통해 받은 기기에 대한 정보를 이용하여,

중재하는 media에 대해 조건적으로 작업을 수행 

특정 조건을 적용할 때 `@대상 + (조건){ 내용 }` 를 사용

```css
<style>
    h1 {
        color: red;
    }
    /* with가 1024px이하 일 때 */
    @media (max-width: 1580px) {
        h1 {
            color: blueviolet;
        }
    }
    /* 대체로 max 보다는 min을 사용하며, '지정 값보다 클 때'에 적용 */
    @media (min-width: 500px) {
        h1 {
            display: none;
            color: darksalmon;
        }
    }
</style>
```

> 주로 사용되는 조건
>
> `min-width: 576px`
>
> `min-width: 768px`

- `ot` : device 방향(세로, 가로 모드)