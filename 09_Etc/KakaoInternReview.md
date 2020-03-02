# Kakao Internship Review

기간: 2019.12.26 - 2020.2.29

부서: 카카오 > 공동체 데이터 센터 > (구)추천팀 (현)머신러닝엔지니어링팀 > 인큐베이터 TF

<br>

### 인턴십을 진행하며 느낀점

- 추천 시스템에 있어서의 통계 지식은 매우 필수적이고 중요한 기본 지식이다.
- 활발한 뇌피셜은 중요하지만, 뇌피셜이 사실임을 확인하는 작업이 훨씬 중요하다.
- 실험은 체계적으로 설계되어야 하며, 설계 내에서의 진행 순서는 실험을 진행하는 이유와 근거다.
- 개발 및 시각화 기술은 기본기이며, 필수적이다.

<br>

<br>

## Projects

- 다음미디어 자동차 동영상 연관 추천
  1. Thompson Sampling, Hyper Parameters Adjustment
  2. Thompson Sampling, Hyper Parameters Adjustment
  3. Ensemble Method, Rank Fusion to Weighted-sum
  4. Ensemble Method, Rank Fusion to Weighted-sum
  5. Item2Vector Model
- 픽코마 작품홈 연관 추천
  1. Word2Vector 개선
  2. Ensemble Method - Weighted-sum to Weighted Rank Fusion

<br>

### Experiment 1. Thompson Sampling, Hyper Parameters Adjustment

실험 기간: 4일

다음미디어의 자동차 동영상의 사용자 시청 로그를 사용하여 생성되는 matrix 데이터가 있었다. 해당 matrix는 ALS를 학습하기 위해 사용되는데, matrix의 sparsity가 다른 서비스들에 비해 매우 높았다. Matrix는 `Users X Items` 형태로 이루어져 있으며, sparsity는 용어의 뜻과는 다르게 `NonZero / User * Item`의 형태로 계산됐다. 즉, 단어 자체는 비어있는 정도에 대해 논하지만, 계산 방식은 matrix에 얼마나 0이 아닌 값으로 채워져 있는가를 의미했다. 따라서 0이 아닌 값들이 다른 서비스들에 많았으므로, 이를 '고인물' 유저가 많다는 것으로 해석했고, 새로운 동영상을 더 많이 보여주는 것이 도움이 될 것이라 생각했다.

새로운 동영상을 더 많이 보여주기 위한 많은 방법들 중, MAB의 empirical draw를 조정하는 것으로 선택했다. Empirical draw는 MAB에서 select 되는 arm들의 일부를 empirical하게 select 한다는 것인데, 이는 Explore가 과하게 진행되는 것을 막기 위해 사용된다. 즉, 비교적 검증된 arm들 중에서만 beta sampling 하여 select하고, 남은 selected에 대해서는 전체 arm들을 대상으로 하여 beta sampling으로 선정한다.

여기서 empirical draw의 대상에 들어가는 threshold를 낮추어 준다면, 비교적 새로운 arm들이 empirical draw의 대상이 될 것이라 판단했고, beta sampling의 기회를 한번 더 갖기 때문에 긍정적인 효과를 가져올 것이라고 판단했다.

하지만 오히려 threshold가 낮아져서 empirical draw의 대상이 된 arm들은 click이 거의 이루어지지 않았고, 덜 검증된 arm들이 오히려 CTR을 하락시켰음을 깨달았다. 추가적으로 결과를 해석해 보자면 다음과 같다.

1. 새로운 arm이라고 해서 그게 새로 업로드된 동영상인 것은 아니다. ALS 혹은 Word2Vector의 알고리즘 변화, 그리고 데이터 변화에 의해 다른 추천 결과가 생성되어 유입된 것일 수도 있다.
2. Sparsity가 높다고 해서 꼭 '고인물'이 많다는 것은 아니다. 이것은 도메인의 특성상, 다른 도메인(서비스)에 비해 유저들이 공통된 동영상(e.g., gc 동영상)을 많이 보는 것을 의미할 수 있다.
3. Exploit이 활발히 이루어지고 있는 상황에서는 이 방법이 유익할 수도 있었겠으나, 트래픽이 적은 현재 도메인 특성상, 위 실험 조건은 지나치게 모험적인 Explore를 촉진시켰다.

<br>

### Experiment 2. Thompson Sampling, Hyper Parameters Adjustment

- 실험 기간: 6일

실험 2에 영향을 미친 실험 1에서 배운 지식은 다음 두 가지이다.

- 다음미디어 자동차 동영상 서비스의 경우 트래픽이 매우 낮으며, 그만큼 추천 결과의 노출 횟수가 매우 적다. 따라서 4일이라는 실험 기간 또한 부족했다.
- Empirical draw 실험의 경우 그 초점이 Exploit에 맞춰져 있었다. MAB의 arm들 중 다수의 arm들이 이미 수렴해 있는 상황에서, 새로운 arm들이 bandit으로 유입될 경우, 빠르게 empirical draw의 대상으로 만드는 것이었다. 하지만 목표하는 Exploit 단계 까지 기다린 후 결과를 비교하는 것은, 적어도 인턴인 우리에게는 시간이 소모적이다.

따라서 Explore에 관여하는 실험을 진행하기로 했고, 그 방법으로 alpha, beta값의 초기값을 조정하는 실험을 했다. 기존 초기값은 지나치게 큰 expected reward를 갖고 있었기 때문에, bandit들은 거의 무조건 새로운 arm들을 select했다. 굉장히 모험적으로 Explore를 진행한 셈이다. 트래픽이 적으며 수렴이 어려운 만큼, 수렴이 될 때까지 Explore를 진행하는 것은 옳지 않다고 판단됐다. 콘텐츠들은 여러 편차들을 가지는데, 그 중 매우 큰 영향력을 가지고 있는 time bias가 있다. 콘텐츠는 시간에 따라 그 가치가 계속해서 변하며, 주로 시간에 따라 감소된다. 따라서 모든 아이템들을 계속해서 Explore하는 것 보다, 아직 새로운 arm들이 많을 지라도 지속적으로 비교적 검증된 arm들을 select 하는 것이 CTR을 향상시킬 것이라 예상했다.

실험 결과 실제로 CTR이 향상됐으며, 대조군과의 비교를 통해 훨씬 적은 비율로 새로운 arm들을 select 하고 있음을 확인했다.

<br>

### Experiment 3, 4. Ensemble Method, Rank Fusion to Weighted-sum

- 실험 기간
  - 실험 3: 4.5일
  - 실험 4: 5일

자동차 동영상이라는 도메인에 익숙한 팀원이 한 명도 없었다. 따라서 추천 결과를 봐도 추천이 잘 되고 있는 것인지에 대한 판단이 서질 않았다. 실제로, ALS를 통해 사용자 소비 패턴을 참고해 보았지만, 도무지 이해가 되지 않았다. 따라서 '차라리 사용자들에게 추천을 맡기자'라고 판단하게 되었고, 앙상블 단계에서 ALS에 힘을 실어주는 실험을 진행하게 됐다.

기존에는 앙상블 기법으로 Rank Fusion을 사용했으며, Rank Fusion은 모델 별 추천 결과들의 순위에 따라 점수를 부여하는 방법이다. Rank Fusion의 경우 모델 별로 모두 동일한 가치를 갖고 있기 때문에, Word2Vector의 1순위 아이템과 ALS의 1순위 아이템이 점수가 같았다. 따라서 Weighted-sum을 사용하여 ALS에 힘을 실어주기로 했고, 이를 위해 Weighted-sum의 적절한 weight set을 찾아야만 했다. Weight set 선정 방법은 다음과 같다.

1. 당시 배포된지 20일 가량이 넘었던, 잘 작동하고 있는 대조군에서 bandit 별로 상위 CTR을 보유하고 있는 arm들을 선정하여 정답 데이터로 삼음
2. Weighted-sum으로 앙상블 했을 때, 정답 데이터들이 가장 많이 포함되는 weight set을 선정

이 때 정답 데이터와 ensemble 결과를 비교할 때에는 Precision과 nDCG를 사용했는데, 여기서는 두 가지 논문을 참고했다.

- [Context-Aware Recommender System: A Review of Recent Developmental Process and Future Research Direction](https://www.researchgate.net/publication/321581444_Context-Aware_Recommender_System_A_Review_of_Recent_Developmental_Process_and_Future_Research_Direction)
- [Performance of recommender algorithms on top-N recommendation tasks](https://www.researchgate.net/publication/221141030_Performance_of_recommender_algorithms_on_top-N_recommendation_tasks)

실험 3은 ALS에 weight이 가중된 파라미터를 사용했으며, 실험 4의 경우 Word2Vector에 힘이 실렸다. 하지만 두 실험 모두 대조군에 비해 낮은 CTR을 기록했으며, 분석한 내용은 다음과 같다.

1. ALS가 overfitting/underfitting 되었을 수 있다.

   이를 파악하기 위해, ALS 학습에 사용되는 파라미터를 확인해 보았으나, alpha/regularization에 해당하는 파라미터의 경우 다른 서비스들의 파라미터와 비교했을 때 유사한 값을 사용하고 있었다. 하지만 latent factor의 dimension이 matrix에서의 item 개수에 비해 지나치게 높다는 것이 확인됐다.

   파라미터를 확인한 후 ALS의 추천 결과 또한 정성적으로 확인해 봤으나, top 100 아이템들이 지나치게 포함되지 않았다는 점(0.3 정도의 비율)에서 overfitting이 되었다고 판단할 수는 없었으며, 되지 않았다고 판단할 수도 없었다.

2. 추천 결과를 생성할 때, 사용되는 아이템들의 리스트가 부적절하다.

   bandit 별로 source 아이템에 대해 추천될 아이템들을 선정하는데, 이 때 추천될 아이템들의 후보군들이 매우 제한되어 있다. 이는 자동차 동영상 연관 추천의 다른 서비스들과의 큰 차이점이다. 해당 추천 아이템 풀의 제한을 없애고 추천 결과를 생성, 아이템 풀의 아이템들이 어느정도 순위에 위치해 있는지 확인해 봤을 때 평균적으로 30위가 넘었다. 즉, ALS와 Word2Vector이 생각하는 진짜 유사한 아이템들은 추천되지 못하고 있었다.

3. Weight set 선정 과정에 잘못된 점이 있다.

   모델 별 추천 결과의 score를 확인하지 못했다. Rank Fusion과는 다르게, Weighted-sum에서는 모델들이 채점한 아이템 별 score를 비교하고, 대소 관계에 따라 순위를 결정한다. 이는 모델들 별 score 1점이 다른 모델들의 1점과 동일한 가치를 지니게 됨을 의미하는데, 모델 별로 score 분포를 확인해 봤을 때 매우 다른 양상을 띄었다.

   만약 score들의 분포를 normalization을 통해 맞춰주거나, weight set 선정 시뮬레이션 당시 weight의 범위를 제한하지 않았더라면, 또 다른 weight set이 선정됐을 수도 있다.

<br>

### Experiment 5. Item2Vector Model

사용중인 아이템 풀이 부적절함을 깨달았지만, 아이템 풀을 계속해서 사용할 수 밖에 없는 상황이기 때문에 이를 최대한 활용하고자 했다. ALS의 경우 user가 시청한 동영상들에 대한 목록을 사용해서 학습을 진행하지만, stream에 대한 정보는 존재하지 않는다. 즉, 어떤 동영상을 봤을 때, 그 직전/직후에 시청항 동영상이 무엇인지는 학습에서 고려하지 않는다. 따라서 이러한 stream 성질을 강화할 수 있는 Item2Vector를 사용할 경우 보다 사용자의 의견을 많이 반영할 수 있을 것이라 생각했다.

Item2Vector의 모델은 이미 라이브러리 존재했기 때문에 이를 활용했으며, 하이퍼 파라미터에서 일반적인 Item2Vector와 큰 차이가 있다. 일반적인 Item2Vector와 논문에서는 window size를 stream 길이 전체(유저 한명이 연달아 시청한 목록)를 사용한다. 실제로 학습후 embedding vector들을 시각화 해 봤을 때, window size가 커짐에 따라 그 cluster들이 분명하게 나뉘는 효과가 있었다. 하지만 이렇게 학습할 경우 우리가 원하는 직전/직후에 대한 정보를 강조하여 학습할 수 없으리라 생각했고, 평균 stream 길이와 중앙값을 고려하여 window size를 선택했다.

Window size 외에, Word2Vector에서 사용하는 frequent sampling을 사용하여 과도하게 자주 등장하는 아이템들을 일부 제외 시킴으로서 gc 아이템들이 과도하게 학습되는 것을 방지했다. 

그 결과 CTR은 소폭이지만 지속적으로 상승했으며, 다음미디어 자동차 동영상 연관 추천에 있어서 ALS 보다 Item2Vector가 더 좋을 수 있음을 확인했다.

<br>

### Experiment 6. Word2Vector 개선

- 실험 기간: 2일

픽코마 작품홈에서는 Word2Vector, ALS, VGG19 모델을 이용하여 text/als/image 측면에서 유사도를 계산한다. 그런데 Word2Vector의 경우 일본어 wikipedia를 데이터셋으로 하여 학습을 했는데, 해당 데이터셋에서도 형태소 분석기를 통해 명사/대명사만을 사용하여 학습했다. 하지만 일본어를 잘 사용하는 팀원이 해당 형태소들만을 사용하는 것에 의문을 가졌고, 일본어 감성어 품사 빈도를 조사한 결과 형용사와 동사의 비중이 38%로 꽤 큰 비중을 차지한다는 연구 결과(Minato, Bracewell, Ren, and Kuroiwa, 2006)를 확인했다.

따라서 명사/대명사/형용사/동사를 포함하여 Word2Vector 모델을 생성하고 활용하는 실험을 진행했지만, CTR과 CVR 모두 하락했다. 이에 대한 분석은 다음과 같다.

1. 감성어 품사의 빈도 중 형용사/동사가 38%이지만, 명사의 경우 47%이다. 심지어 な형용사의 경우 명사로 분류되며, 추천에 필요한 감성어는 명사로도 충분히 충당되고 있을 수 있다.
2. 형태소 분석 시 문법저긍로 존경, 수동을 뜻하는 (ら)れる가 동사로 분류된다. 즉, 동사를 포함할 경우 실질적 의미가 없는 형태소들이 포함되어 임베딩이 정확하게 이루어지지 않았을 수 있다.
3. 현재 형태소 분석기는 분류 성능이 떨어지므로, 형태소 분석기를 개선시킨 후에 Word2Vector을 조정하는 것이 맞을 수 있다.

<br>

### Experiment 7. Ensemble Setting, Weighted-sum to Weighted Rank Fusion

- 실험 기간: 3일

픽코마 작품홈에서는 철저하게 ALS 위주의 추천을 진행한다. 앙상블 결과 리스트의 모든 아이템들은 ALS를 사용한 추천 결과에 존재하는 아이템들이며, Word2Vector와 VGG19 모델들은 ALS 리스트의 아이템들에 가중치를 부여하여 그 리스트를 재정렬하는 효과로 사용된다. 이러한 추천 형태는 empirical하게 실험해 봤을 때 CTR과 CVR이 가장 높게 나타나기 때문에 사용되고 있다.

그런데, 이전 실험 기록을 통해 ALS만 사용하여 추천하는 것이 가장 높은 CVR을 보이고 있음을 확인했으며, Word2Vector와 VGG19는 CF 알고리즘인 ALS의 cold start problem을 완화하기 위해 사용되고 있었다. 하지만 현재처럼 Weighted-sum을 사용할 경우 Word2Vector와 VGG19가 ALS 추천 결과의 순서를 크게 좌우하게 됨을 알 수 있었다. 예를 들자면, `(20위의 ALS 추천 아이템)`이, `(80위의 ALS 아이템 + 80위의 Word2Vector 아이템)` 조합에 의해 밀려나는 현상을 쉽게 확인할 수 있었다. ALS 위주의 추천을 위해 사용하는 Weighted-sum이지만, 부분적인 중복에 의해 순위 조정이 너무 크게 이루어지고 있다고 판단했다.

이러한 문제를 해결하고자 Weighted Rank Fusion을 사용해 보았다. Rank Fusion은 모델 별 순위에 따라 점수를 부여하여 앙상블을 진행하는데, 그냥 Rank Fusion을 사용할 경우 모든 모델들이 같은 가치를 지니게 된다. 즉, ALS 2순위 아이템이 Word2Vector 혹은 VGG19에서의 1순위 아이템보다 낮은 순위를 갖는다. 따라서 ALS 위주의 추천을 유지하기 위해 Weighted Rank Fusion을 사용했다. Weighted Rank Fusion은 말그대로 순위 + 가중치를 사용하는 것이며, k 값이 작아질 수록 높은 순위와 낮은 순위의 점수 차이가 벌어지며, 높아질 수록 그 차이가 작아진다. 따라서 `k` 값이 너무 커질 경우, weight까지 계산 했음에도 불구하고 ALS의 저순위 아이템이 Word2Vector/VGG19의 고순위 아이템보다 점수가 낮아지는 현상이 발생하므로, ALS 위주의 추천이 일부 적용되지 않는 현상이 있다. 따라서 ALS의 순위를 너무 파괴하지 않으며, ALS에 존재하는 아이템들만을 추천하도록 하는 하이퍼 파라미터 `k` 값을 찾아 실험했다.

하지만 실험 결과 CTR과 CVR 모두 하락했다. 결과 분석을 하기 전에 설명하고자 하는 Rank Fusion에서의 세 가지 고려사항은 다음과 같다.

- Skimming Effect

  각 모델 별 상위 순위의 아이템들을 더 고려함

- Dark Horse Effect

  각 모델들 중 특별히 잘 예측하는 모델이 있고, 그 모데렝 weight을 부여

- Chorus Effect

  각 아이템들이 모델 별 리스트에 얼마나 많이 등장하는가를 최종 점수 계산에 고려

이 세 가지는 Rank Fusion을 사용함에 있어 고려해야 하는 내용이며, 논문에 언급되어 있다. 현재 우리가 실험하는 내용은 Chorus Effect를 약화시키고, Dark Horse Effect를 강화시키는 내용이다. 하지만 실험 결과의 낮은 지표들을 고려했을 때, 그리고 유사한 상황에서의 연구 결과(Vogt & Cottrell, 1998)를 참고했을 때 Chorus Effect는 실제로 중요한 역할을 하고 있다고 판단된다.