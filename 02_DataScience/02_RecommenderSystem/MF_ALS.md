# MF & ALS

출처: [Y.LAB](https://yamalab.tistory.com/89)

이 글은 Y.LAB님의 게시글의 일부를 그대로 퍼온 것입니다.

<br>

## MF, Matrix Factorization

추천 알고리즘 중에 가장 널리 사용되는 알고리즘은 단연 CF이다. 그중에서도 단일 알고리즘으로써 가장 성능이 높은 것은 단연 Model Based CF인 Matrix Factorization 계열의 알고리즘이다. ([CF 기반 모델링에 관한 내용 참고](http://yamalab.tistory.com/69)) 이번 포스팅에서는 MF와 ALS에 대해 공부해보고, Implicit feedback 상황에서 ALS를 적용하는 방법을 살펴볼 것이다. 포스팅의 내용은 [추천 엔진을 구축하기 위한 기본서 - Suresh K.Gorakala]와 [논문](http://yifanhu.net/PUB/cf.pdf) 그리고 [이곳](http://sanghyukchun.github.io/95/)을 참고하였다.

우선 MF 모델은 user-item 의 matrix에서 이미 rating이 부여되어 있는 상황을 가정한다. (당연히 sparse한 matrix를 가정한다) MF의 목적은, 바꿔 말하면 Matrix Complement 이다. 아직 평가를 내리지 않은 user-item의 빈 공간을 Model-based Learning으로 채워넣는 것을 의미한다.

유저간, 혹은 아이템간 유사도를 이용하는 Memory-based 방법과 달리, MF는 행렬 인수 분해라는 수학적 방법으로 접근한다. 이는 행렬은 두개의 하위 행렬로 분해가 가능하며, 다시 곱해져서 원래 행렬과 동일한 크기의 단일 행렬이 될 수 있다는 성질에 기인한 것이다.

크기 $U \times M$을 가지는 rating matrix $R$이 있다고 하자. 이 때 R은 각각의 크기가 $U \times K$, $M \times K$인 두 개의 행렬 $P$와 $Q$로 분해될 수 있다고 가정해본다. 그리고 다시 $P \times Q$ 행렬을 계산하면, 원래의 matrix $R$와 굉장히 유사하며 크기가 동일한 행렬이 생성된다. 중요한 것은, 행렬이 재생성 되면서 빈공간이 채워진다는 것이다. 이러한 행렬 인수 분해의 원칙은 비평가 항목을 채우기 위함이라는 것을 알 수 있다. 또한 한 가지 알 수 있는 것은, 분해된 행렬 $P$, $Q$는 각각 User-latent factor matrix, Item-latent factor matrix라는 각각의 내재적 의미를 나타내는 잠재 행렬도 나타낼 수 있다는 것이다. 이는 사람이 해석하는 잠재의미로도 해석은 가능하지만 기계가 해석하기 위한 행렬, 즉 블랙 박스 모델에 더 가깝다.

<br>

### Objective Function

이제 MF를 학습하는 것은 latent feature들을 학습하는 것과 같다는 것을 알게 되었다. Latent 행렬을 각각 $P$, $Q$라고 했을 때 이제 MF 모델의 목적함수는 다음과 같다.

$min_{q, p} \; \sum_{(u,i)\in K} \; (r_{ui}-q^{T}_{i}p_{u})^2 + \lambda (||q_{i}||^2 + ||p_{u}||^2)$

이 목적함수를 최소화 하는 것이 $P$와 $Q$를 학습하기 위한 것이다. 결국 rating의 $(y-\hat{y})^2$ 을 오차로 활용하는 것이기 때문에, 일반적인 regression에서의 최적화와 마찬가지로 정규화 파라미터를 추가해준다. 

<br>

<br>

## ALS, Alternating Least

교대 최소 제곱법, ALS는 위에서 정의한 목적함수를 최적화하는 기법이다. 일반적인 파라미터 최적화 기법으로 Gradient Descent를 사용하지만, 추천 알고리즘의 computation 환경인 분산처리 플랫폼에서는 GD보다 ALS가 더욱 효과적이라고 알려져 있다. 이는 ALS의 계산 방법 때문인데, GD에서처럼 Loss에 대한 편미분값을 update gradient로 활용하는 것이 아닌, $P$와 $Q$ 벡터 중 하나를 고정해놓고 교대로 계산하기 때문이다. 이러한 방법은 분산처리 환경에서 더욱 빠른 연산이 가능해진다. 또한 ALS는 GD와 비교해볼때 sparse한 데이터에 robust한 모습을 보인다고 알려져 있다. Spark에서는 MF의 기본 학습법을 ALS에 기반하여 ML library로 제공하는데, Implicit feedback에 최적화된 학습까지 제공한다. ([논문](http://yifanhu.net/PUB/cf.pdf))