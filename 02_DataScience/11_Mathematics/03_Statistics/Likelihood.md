# Likelihood

참고자료: [Chris AI Holy](https://youtu.be/mxCmB1WE3R8)



## 우도, Likelihood

어떤 값을 갖을 때, 해당하는 분포가 얼마나 값을 설명하는가, 분포가 얼마나 값에 적당한가 $P(분포 | 값)$

<br>

### 크로스 엔트로피, Cross Entropy

크로스 엔트로피는 classification 모델의 loss function으로 가장 많이 사용되는 함수입니다. 크로스 엔트로피는 두 분포(Distribution)의 차이를 구하는 함수이며, 두 분포가 어떤 분포를 따르는가를 고려하지 않으며, 단순히 얼마나 차이가 나는가를 수치화하는 함수입니다.

우리는 머신러닝 모델을 학습시킬 때, 우도를 통해 현재 $\theta$ 들의 분포를 추정한 후, 실제 데이터(Train dataset, observed data)의 분포와의 차이를 구하여 이를 손실(Loss) 및 비용(Cost)으로 사용합니다.

