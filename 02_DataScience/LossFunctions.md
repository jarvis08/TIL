# Loss Functions

- 참고 자료
  - [seongkyun](https://seongkyun.github.io/study/2019/04/18/l1_l2/)
    - L1 Loss, Regularization
    - L2 Loss, Regularization
  - [ratsgo]([https://ratsgo.github.io/deep%20learning/2017/09/24/loss/](https://ratsgo.github.io/deep learning/2017/09/24/loss/))
    - Negative Log-likelihood
  - [TowardsDataScience](https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23)
    - Loss Functions

<br>

<br>

## Definition of loss function

[위키백과]([https://ko.wikipedia.org/wiki/%EC%86%90%EC%8B%A4_%ED%95%A8%EC%88%98](https://ko.wikipedia.org/wiki/손실_함수))에서는 손실 함수를 다음과 같이 정의한다.

> [통계학](https://ko.wikipedia.org/wiki/통계학), [결정이론](https://ko.wikipedia.org/wiki/결정이론) 및 [경제학](https://ko.wikipedia.org/wiki/경제학) 분야에서 **손실 함수**(損失函數) 또는 **비용 함수**(費用函數)는 [사건](https://ko.wikipedia.org/wiki/확률공간)(기술적으로 [표본 공간](https://ko.wikipedia.org/wiki/표본_공간)의 한 요소)을 그 사건과 관련된 경제적 손실을 표현하는 [실수](https://ko.wikipedia.org/wiki/실수)로 사상하는 함수이다.

머신러닝에서의 손실 함수는 데이터셋의 입력값이 주어졌을 때, 입력값에 대한 정답을 반환할 수 있도록 모델의 가중치들을 학습시키는데에 사용된다. 손실 함수는 가중치들의 집합인 $\theta$ 를 사용하여 값을 예측하며, 실제값과 예측값의 차이를 계산하여 반환한다. 모델은 손실 함수의 결과값을 최소화 하는 방향으로 $\theta$ 를 수정한다.

<br>

### Loss functions

Broadly, loss functions can be classified into two major categories depending upon the type of learning task we are dealing with — **Regression losses** and **Classification losses**. In classification, we are trying to predict output from set of finite categorical values i.e Given large data set of images of hand written digits, categorizing them into one of 0–9 digits. Regression, on the other hand, deals with predicting a continuous value for example given floor area, number of rooms, size of rooms, predict the price of room.

<br>

<br>

## In Regression.

**산술 손실 함수 - 산술값을 예측할 때 데이터 대한 예측값과 실제 관측 값을 비교하는 함수 (regression)**

회귀식에서 사용되는 손실 함수들은 다음과 같다.

- MBE, Mean Bias Error
- MSE, Mean Square Error
- MAE, Mean Absolute Error

<br>

### Mean Bias Error

$MBE = \frac{\sum^{n}_{i=1}(y_{i} - \hat{y}_{i})}{n}$

This is much less common in machine learning domain as compared to it’s counterpart. This is same as MSE with the only difference that we don’t take absolute values. Clearly there’s a need for caution as positive and negative errors could cancel each other out. Although less accurate in practice, it could determine if the model has positive bias or negative bias.

<br>

### Mean Square Error / Quadratic Loss / L2 Loss / Least square error(LSE)

$MSE = \frac{\sum^{n}_{i=1}(y_{i} - \hat{y}_{i})^2}{n}$

As the name suggests, *Mean square error* is measured as the average of squared difference between predictions and actual observations. It’s only concerned with the average magnitude of error irrespective of their direction. However, due to squaring, predictions which are far away from actual values are penalized heavily in comparison to less deviated predictions. Plus MSE has nice mathematical properties which makes it easier to calculate gradients.

손실 값이 제곱되어 더해지므로, 노이즈가 포함될 경우 L1 Loss 보다 학습에 영향을 많이 받는다.

<br>

### Mean Absolute Error / L1 Loss / Least Absolute Deviations(LAD)

$MAE = \frac{\sum^{n}_{i=1}|y_{i} - \hat{y}_{i}|}{n}$

*Mean absolute error*, on the other hand, is measured as the average of sum of absolute differences between predictions and actual observations. Like MSE, this as well measures the magnitude of error without considering their direction. Unlike MSE, MAE needs more complicated tools such as linear programming to compute the gradients. Plus MAE is more robust to outliers since it does not make use of square.

<br>

### L1, L2 Regularization

- L1 Regularization

  $cost(W, b) = \frac{1}{m}\sum^{m}_{i}L(\hat{y}_{i}, y_{i}) + \lambda \frac{1}{2}|w|$

- L2 Regularization

  $cost(W, b) = \frac{1}{m}\sum^{m}_{i}L(\hat{y}_{i}, y_{i}) + \lambda \frac{1}{2}|w|^{2}$

<br>

## L1 & L2 Loss

### Robustness(L1 > L2)

- Robustness는 outlier, 즉 이상치가 등장했을 때 loss function이 얼마나 영향을 받는지를 뜻하는 용어
- L2 loss는 outlier의 정도가 심하면 심할수록 직관적으로 제곱을 하기에 계산된 값이 L1보다는 더 큰 수치로 작용하기때문에 Robustness가 L1보다 적게된다.
  - 제곱의 합이므로 당연히 더해진 값이 더 크다.
- 따라서 outliers가 효과적으로 적당히 무시되길 원한다면 비교적 이상치의 영향력을 작게 받는 L1 loss를, 반대로 이상치의 등장에 주의 깊게 주목을 해야할 필요가 있는 경우라면 L2 loss를 취하여야 한다.

<br>

### Stability(L1 < L2)

- Stability는 모델이 비슷한 데이터에 대해 얼마나 일관적인 예측을 할 수 있는가로 생각하면 된다. 이해를 위해 아래의 그림을 보자.

![views](https://seongkyun.github.io/assets/post_img/study/2019-04-18-l1_l2/fig1.gif)

- 위 그림에서 실제 데이터는 검은 점으로 나타난다.
- 위 그림에서 실제 데이터(검은점)와 Outlier point인 주황색 화살표의 점이 움직임에 따라 어떻게 각 L1과 L2에 따라 예측 모델이 달라지는지를 실험해 본 결과이다.
- Outlier point가 검은 점들에 비교적 비슷한 위치에 존재할 때 L1 loss 그래프는 변화가 있고 움직이지만, L2 loss 그래프에는 그러한 변화가 없다. 이러한 특성때문에 L1이 L2보다는 unstable하다고 표현한다.
- 위 그림에서 또한 robustness도 관찰 가능한데, outlier point가 검은점들의 경향성이 이어지는 선을 기준으로 왼쪽이서 오른쪽으로 이동할 때 L2 error line이 L1보다 더 먼저 움직이는것을 확인 할 수 있다.
  - 즉, L1보다 L2가 먼저 반응하므로 L1이 robust하고, outlier의 움직임에 L2보다 L1이 더 많이 움직이기에 L2가 stable하다.

<br>

<br>

## In Classification.

**확률 손실 함수 - 특정 항목을 나누는 분류에서 사용**

분류 모델에서 사용되는 대표적인 손실 함수들은 다음과 같습니다.

- Hinge Loss
- CEE, Cross Entropy Loss

<br>

### Hinge Loss/Multi class SVM Loss

$SVMLoss = \sum_{j≠y_{i}}max(0, s_{j} - s_{y_{i}}+1)$

In simple terms, the score of correct category should be greater than sum of scores of all incorrect categories by some safety margin (usually one). And hence hinge loss is used for [maximum-margin](https://link.springer.com/chapter/10.1007/978-0-387-69942-4_10) classification, most notably for [support vector machines](https://en.wikipedia.org/wiki/Support_vector_machine). Although not [differentiable](https://ipfs.io/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Differentiable_function.html), it’s a convex function which makes it easy to work with usual convex optimizers used in machine learning domain.

<br>

### Cross Entropy Loss(CEE) / Negative Log Likelihood

$CrossEntropyLoss = -\sum_{x}P(x)logQ(x)$

만약 분류하는 label 및 class가 0과 1로 이루어진 binary 형태라면, 다음과 같이 표현할 수 있다.

$CrossEntropyLoss = -(y_{i}log\hat{y}_{i} + (1-y_{i})log(1-\hat{y}_{i}))$

<br>

### Why negative?

딥러닝 모델의 손실함수로 **음의 로그우도(negative log-likelihood)**를 사용한다. 

"손실함수로 음의 로그우도을 쓸 경우 몇 가지 이점이 생긴다고 합니다. 우선 우리가 만드려는 모델에 다양한 확률분포를 가정할 수 있게 돼 유연하게 대응할 수 있게 됩니다. 음의 로그우도로 딥러닝 모델의 손실을 정의하면 이는 곧 두 확률분포 사이의 차이를 재는 함수인 크로스 엔트로피가 되며, 크로스 엔트로피는 비교 대상 확률분포의 종류를 특정하지 않기 때문"

<br>

<br>

## Why Regression-MSE / Classification-CEE

<br>

<br>

## 기타

**랭킹 손실 함수 - 모델이 예측해낸 결과값에 순서가 맞는지만 판별**

- pairwise zero-one - 관계가 잘못된 경우를 카운팅
- edit distance - 몇 번의 맞바꿈을 해야 원래 순서로 돌아갈 수 있는지 측정