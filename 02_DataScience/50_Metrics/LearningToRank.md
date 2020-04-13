# Learning to Rank

[TOC]

---

## Introduction

Learning to Rank란 ranking task에서 모델을 학습시키기 위한 머신러닝 방법이고, Ranking task는 ranking model $f(q, d)$를 이용해 query $q$에 대한 document들을 정렬하게 하는 task이다. 머신러닝이 발전하면서, $f(q, d)$를 머신러닝을 이용해 추정할 수 있게 되었다.

### Traning Data

Learning to rank는 supervised learning task이기 때문에, training data는 어떻게 구성되어 있는지 알아볼 필요가 있다.

Training data는 query들과 document들로 이루어져 있다. 각각의 query들은 몇몇의 documents와 관련이 되어있고(query마다 다를 수 있음), 그 query에 대한 documents의 relevance 또한 주어져 있다. 이러한 relevance를 label이라고 말하고 label들은 몇 단계의 grade (level)로 나타내진다. 높은 grade일수록 query와 document가 더 관련되어 있다고 말할 수 있다.

#### Denotation

- $Q$: the query set
- $D$: the document set
- $Y = \{1, 2, \cdots, l\}$: the label set, where labels represent grades
- $\{q_1, q_2, \cdots, q_m\}$: the set of queries for traning
    - $q_i$: the $i$-th query
- $D_i = \{d_{i, 1}, d_{i, 2}, \cdots, d_{i, n_i}\}$: the set of documents associated with query $q_i$
    - $n_i$: the sizes of $D_i$
    - $d_{i, j}$: $j$-th documents in $D_i$ 
- $Y_i = \{y_{i, 1}, y_{i, 2},\cdots, y_{i, n_i}\}$: the set of labels associated with query $q_i$
    - $y_{i,j}$: $j$-th grade label in $Y_i$, representing the relevance degree of $d_{i, j}$ with respect of $q_i$
- $S = \{(q_i, D_i), Y_i\}^m_{i=1}$: the original training set

이렇게 표현되어 있는 공간에서 우리는 feature vector를 고려할 수 있다.

- $x_{i,j} = \phi (q_i, d_{i,j})$: the feature vector
    - $(q_i, d_{i,j})$: each query-document pair, where
        - $i = 1, 2, \cdots, m$
        - $j = 1, 2, \cdots, n_i$
    - $\phi$: the feature function
- $X_i := \{x_{i, 1}, x_{i, 2}, \cdots, x_{i, n_i}\}$, 만약 feature vector들을 다음과 같이 표현한다면,
- $S' = \{(X_i, Y_i)\}^m_{i=1}$: training data set을 이렇게 표현할 수 있다.

**Ranking model** $f(q, d)$는 주어진 query-document pair $q$ & $d$에 대해, 즉 feature vector $x$에 대해 ranking model 하나의 적절한 실수 값을 $f(q, d) (= f(x))$에 할당하는 함수이다.

이 $f(x)$를 이용하여, 주어진 query $q_i$와 documents $D_i$에 대해 $D_i$의 원소를 적절한 순서로 배치하는 것을 **Ranking**이라 할 수 있다.

**추론 단계**에서는 새로운 query $q_{m+1}$과 associated documents $D_{m+1}$에 대해 feature vector $X_{m+1}$을 만들고 ranking model $f(x)$를 통해 score를 할당하며 이를 이용해 $D_{m+1}$를 정렬한다.

---

## Learning to Rank's Evaluation

[출처](https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf)

'한 query에 대해 모델이 출력한 ranking이 잘 되었는가'를 평가하기 위해서는 우리가 사용하던 metric과는 다른 metric을 사용해야 한다.

### Multi-label Metric, P@K, R@K(Precision at K, Recall at K)

- P@K: proportion of top-K documents that are relevant
    - [top-K 문서들] 중 [query와 관련이 있는 문서들]의 비율
- R@K: proportion of relevant documents that are in the top-K
    - [query와 관련이 있는 문서들] 중 [top-K에 있는 문서들]의 비율

[p16](https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf)의 예시를 따라가보자!

#### P/R@K의 장단점

- 장점
    - 계산하기 쉽다
    - 해석하기 쉽다
- 단점
    - K가 metric에 큰 영향을 준다
    - top-K 안에 있는 문서의 순서들의 ranking은 중요하지 않다.
    - K를 선택해야한다.

### Multi-label Metric, AP(Average-Precision)

Average precision은 p@k와 r@k를 k의 선택없이 모두 고려하고자 하는데서 나온 metric이다.

AP를 구하는 방법은 다음과 같다.

1. Go down the ranking one-rank-at-a-time
2. If the document at rank K is relevant, measure P@K
3. Finally, take the average of all P@K values

예시를 보자. [p35](https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf)

#### AP의 장단점

- 장점
    - K를 선택할 필요가 없다
    - p@k와 r@k를 모두 고려한다.
    - top에서의 ranking 실수가 ap에 영향을 끼친다.
    - bottom에서의 ranking 실수가 ap에 영향을 끼친다.
        - top에서의 실수가 bottom에서의 실수보다 더 크게 영향을 끼친다.
- 단점
    - p/r@k를 쉽게 해석할 수 없다.

### Learning to Rank's Evaluation, MAP, DCG, NDCG

#### MAP, Mean Average Precision

이제까지 우리는 single query에 대해서만 논했었다. 한 데이터셋에 대해 모델의 전체적인 성능을 파악하려면 모든 query에 대한 AP값을 구해 평균을 내야한다. 이를 MAP(mean average precision)라고 한다.

#### DCG, discounted-cumulative gain

p@k, r@k, AP까지는 binary relevance만 고려하였다(관련도가 0 아니면 1).
$$
DCG@K = \sum^{K}_{i=1}\frac{REL_i}{\operatorname{log}_2(max(i,2))} \\
\text{where } REL_i \text{ is the relevance grade. }(\{0, 1, 2, 3, \dots\})
$$
여기서 $log$가 나오는 이유는 relevant document의 유용성은 rank가 증가할 때마다 기하적으로 떨어짐을 반영하고자이다.

#### NDCG, Normalized discounted-cumulative gain

DCG에서 주의할 점은 K값이 커지면 커질 수록 DCG값도 같이 커진다. 즉, K가 엄청나게 크다면 metric값도 한없이 커지므로 모델에 대한 정확한 평가방법이라고 할 수는 없다.

여기서 NDCG 개념이 나오는데 NDCG는 다음과 같이 구한다.

- 주어진 query에 대해, DCG를 측정한다.
- 이 DCG 값을 best possible DCG값으로 나눈다.

---

## Learning to Rank's Loss Functions

이제 모델의 평가 방법을 알았으니 Loss 함수를 정의하고 gradient descent를 통해 파라미터들을 조정해야한다.

Learning to Rank의 true loss는 다음과 같이 정의할 수 있다.
$$
Loss = 1 - MAP
$$

$$
Loss = 1 - DCG
$$

$$
Loss = 1 - NDCG
$$

하지만 이 값은 당연하게도 미분가능하지 않다. 심지어 연속함수도 아니다. 따라서 차선책으로, 이 값들을 upper bound하는 함수들을 이용해 이 값을 minimize한다. 이 upper bound function들은 형태에 따라 3가지로 구분된다.

- Pointwise Loss
- Pairwise Loss
- Listwise Loss

각각의 대표 loss들을 알아보도록 하자.

- 참고, tensorflow-ranking package에 있는 loss들
    - pointwise loss
        - sigmoid cross entropy
        - mean squared loss
    - pairwise loss
        - pairwise hinge loss
        - pairwise logistic loss
        - pairwise soft zero one loss
    - listwise loss
        - softmax loss
        - list mle loss
        - approx ndcg loss
        - approx map loss
        - gumbel approx ndcg loss
        - neural sort cross entropy loss
        - gumber neural sort cross entropy loss

### Pointwise Loss

Pointwise Loss는 각 class에 대한 true값과 예측값만 loss에 영향을 끼친다.

- sigmoid cross entropy
    $$
    l(y, \hat{y}) = - \sum^{n-1}_{j=0} y_i log(\hat{y}_j) + (1 - y_j) log(1-\hat{y}_j)
    $$

### Pairwise Loss

pairwise loss는 두 개의 class에 대한 쌍을 관찰하며 loss를 구한다.

- pairwise hinge loss
    $$
    l(y, \hat{y}) = \sum^{n-1}_{i=0}\sum^{n-1}_{j=0}\mathbb{I}(y_i=0\&y_j=1)max(0, \alpha + \hat{y}_i - \hat{y_j})
    $$


### Listwise Loss

Listwise loss는 수식이 더 복잡하고 어렵다.

- [approx map loss](https://arxiv.org/pdf/1906.07589.pdf)
    $$
    mAP_Q(D, Y) = \frac{1}{B} \sum^{B}_{i=1}AP_Q(d^T_iD, Y_i) \\
    \text{ where } AP_Q(S^q,Y^q) = \sum^{M}_{m+1}\hat{P}_m (S^q, Y^q)\triangle \hat{r}_m(S^q, Y^q).
    $$

#### 이러한 loss들이 위의 true loss를 bounded 한다는 증명은?

이 증명은 매우 어려워보인다… 저자를 믿는 수 밖에…

https://papers.nips.cc/paper/3708-ranking-measures-and-loss-functions-in-learning-to-rank.pdf

#### 대신 우리는 예시를 통해 이러한 loss가 합리적인가를 알아보자.

$q = ``\text{I'm so happy and excited}"$

$label = \{0: ``happy", 1: ``sad", 2: ``excited"\}$

$y = (1, 0, 1)$

$\hat{y} = (0.5, 0.3, 0.2)$