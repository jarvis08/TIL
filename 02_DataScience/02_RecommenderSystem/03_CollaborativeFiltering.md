# Collaborative Filtering

Categories of CF techniques:

- Memory-based
- Model-based
- Hybrid CF Algorithms (= CF + Older recommendation techniques)

<br>

<br>

## Memory-based CF

유사도를 기반으로 유사한 유저를 탐색하고, 유사한 유저들이 평가한 명시/암시적인(explicit/implicit) 평가 점수들을 토대로 타겟 유저의 평가 점수를 예측하는 방법입니다.

아마존에서 또한 변형된 Memory-based CF를 사용했으며(현재는 어떻게 사용하는지 검색해보지 않았음), 그 이유는 구현이 쉽고, 유저들의 아이템 탐색 시간을 감소시켜 충성도를 높이며, 기업의 수입과 광고 수입을 증가시켜주기 떄문입니다.

<br>

### Limitations of Memory-based CF

하지만 많은 장점이 존재하는 memory-based CF에도 문제점이 존재합니다. 유사도 값들이 유저들 간의 공통된 아이템에 의해 계산되므로, 데이터 자체가 거의 존재하지 않거나 공통 데이터가 없는 경우 희소 데이터(**sparse data**) 문제에 취약합니다.

<br>

<br>

## Model-based CF

Memory-based CF의 단점을 극복하고, 보다 개선된 예측 성능을 얻기 위해 만들어진 것이 바로 model-based CF입니다. Model-based CF는 모델이 점수를 예측하도록 학습 혹은 평가할 때 오직 순수한 **rating data**만을 사용합니다. 모델은 데이터 마이닝, 머신러닝과 같은 알고리즘들을 말합니다. 잘 알려진 model-based CF 기술들은 다음과 같습니다.

- BNs, Bayesian belief Nets CF models
- Clustering CF models
- Latent Semantic CF models
- MDP(Markov Decision Process)-based CF system

<br>

<br>

## Hybrid CF techniques

Hybrid CF 기술들은 기존 추천 시스템 기술들과 CF를 결합한 형태를 말합니다. 이전 기술들의 예로 **Content-based Filtering**이 있는데, content-based filtering과 CF의 가장 큰 차이점은 CF는 오직 유저들이 아이템에 대해 평가한 rating data만을 사용하며, content-based filtering은 아이템과 유저의 자체적인 특징(feature) 데이터를 사용한다는 것입니다. 그리고 이러한 특징들로 인해 두 기술들의 단점들 또한 나뉩니다. CF의 경의 명시적으로 특징 정보를 보유하지 않으며, content-based system들은 유저간의 선호 유사도에 대한 정보를 고려하지 않습니다.

<br>

### Hybrid CF

아래 두 알고리즘은 위에서 언급된 문제들을 회피하기 위해 CF와 content-based filtering을 결합한 알고리즘입니다.

- Content-based CF algorithm
- Personality Diagnosis

<br>

<br>

## Etc

### Evaluation

CF 알고리즘들을 평가하기 위해서는 CF application에 맞는 metric을 설정해야 합니다. 그러기 위해서는 classification error 대신, 가장 널리 사용되는 예측 성능의 평가 지표는 MAE(Mean Absolute Error)입니다.

Precision과 recall은 information retrieval research에서 ranked list들의 아이템들을 반환하는 알고리즘의 지표로 널리 사용됩니다.

ROC sensitivity는 decision support의 정확도 지표로 자주 사용됩니다.

<br>

### Datasets

인위적인 데이터를 통해서는 확신을 주는 결론을 도출하기에는 위험이 있으므로, CF 연구에서는 현실의 실험적인 데이터를 사용하는 것이 바람직합니다. 주로 사용되는 CF database들은 다음과 같습니다.

- MovieLens
- Jester
- Netflix prize data

