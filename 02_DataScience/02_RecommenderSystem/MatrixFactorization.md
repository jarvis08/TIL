# Matrix Factorization

## Recommender System

추천 시스템은 크게 두 가지로 나뉜다.

- Content Filtering
- Collaborative Filtering

Content filtering은 주로 content-based recommendation이라고 부르며, item의 정보 혹은 user의 정보를 통해 추천할 item 혹은 user를 선정한다. Content-based recommendation은 연관 추천에서 자주 사용된다. 연관 추천은 user에 상관없이 item 간의 유사성과 같은 지표를 생성하여 추천에 활용한다.

Collaborative filtering은 user와 item의 정보를 모두 사용하여 추천할 item 혹은 user를 선정한다. 대표적인 예시로 user가 평가한 item들을 이용하여 평가하지 않은 다른 item에 대한 rating을 예측하는 것이 있다. Collaborative filtering은 주로 개인화 추천에 사용된다.

<br>

### Collaborative Filtering

Collaborative filtering은 다시 두 가지로 나뉜다.

- Neighborhood methods
- Latent factor models

위에서 설명한 예시가 바로 neighborhood methods이다. `user_1`이 rating한 item들을 토대로 `user_1`과 유사한 또 다른 `user_2`를 찾는다. 그 후, 찾아낸 비슷한 `user_2`의 rating들을 토대로 `user_1`이 경험해보지 못한, `user_1`이 높게 rating할 것이라 예측되는 item들을 추천한다.

Latent factor model은 user들이 rating한 내용들을 토대로 user들의 특성을 나타내는 matrix와 item들의 특성을 나타내는 matrix로 나눈다. 즉, neighborhood method들과 같이 유사도를 통해 값을 예측하는 것이 아니라, user와 item의 고유한 feature값을 생성(예측)한다.

<br>

### CF의 단점

- `Cold Start`
  - New user problem
  - New item problem
- `Sparse Data`: rating 내역이 충분하지 않은 경우
- `Gray/Black Sheep`: rating한 내용이 특정 취향을 가려낼 수 없는 경우
- `Shilling Attack`: 고의/의도적으로 특정 아이템의 rating을 조작하는 행위

<br>

<br>

## Matrix Factorization

