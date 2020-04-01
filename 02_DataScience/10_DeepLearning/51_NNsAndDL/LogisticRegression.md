# Logistic Regression

Logistic Regression과 Linear Regression은 독립 변수들의 선형 결합으로 종속 변수의 값을 예측합니다. 하지만 Logistic Regression은 범주형 데이터를 예측하므로 classification 문제에 사용되며, Linear Regression은 연속형 데이터를 예측하므로 주가 예측과 같은 문제에 사용합니다.

<br>

### Notations

아래 그림은 고양이 사진을 3개의 channel로 나눈 후, 이를 열 벡터 데이터로 표현하는 방법에 대한 내용입니다.

![VectorOfImageData](./assets/VectorOfImageData.png)

3개 channel 데이터들을 하나의 열 벡터로 표현하는 것은 행렬곱으로 계산을 편하게 하기 위함입니다. 아래 그림은 기타 수식의 notation을 정리한 그림입니다.

![Notations](./assets/Notations.png)

m개의 사진들에 대한 데이터들을 하나의 `X` 행렬로 표시합니다. `X`의 원소인 $x^{(1)...(m)}$은 각각 $n_x$ 차원으로 나타내고 있는 그림들입니다.

![ComputationGraph](./assets/ComputationGraph.png)

![ComputingDerivatives-1](./assets/ComputingDerivatives-1.png)

![CoputingDerivatives-0](./assets/CoputingDerivatives-0.png)

![CostFunctionOfLogisticRegression](./assets/CostFunctionOfLogisticRegression.png)

![GradientDescent-0](./assets/GradientDescent-0.png)

![GradientDescent-1](./assets/GradientDescent-1.png)

![LogisticRegression](./assets/LogisticRegression.png)

![LogisticRegressionDerivatives](./assets/LogisticRegressionDerivatives.png)

![LogisticRegressionOn-M-Examples](./assets/LogisticRegressionOn-M-Examples.png)

