# 05_MAB, Multi-armed Bandit

여러개의 슬롯머신들이 있을 때, **제한된 시도 횟수를 보유한 상태에서 최대한 보상을 많이 받아내는 방법**에 대한 문제를 **MAB Problem**이라고 합니다. 여기서 머신들 별 보상 확률들을 알아내는 과정을 **탐색(Explore)**이라고 하며, 탐색해 낸 보상 확률들을 토대로 보상을 취하려 하는 행위를 **활용(Exploit)**이라고 합니다.

MAB(Multi-armed Bandit)는 Netflix를 예를들어 생각해 보면 그 구조를 이해하기 쉽습니다. Bandit은 source 데이터를 의미하며, Netflix에서는 현재 시청중인 드라마 혹은 영화를 의미합니다. 이름에서 알 수 있듯이, bandit 하나에 여러 개의 arm들이 존재합니다. 여기서의 arm 하나 하나는 source 데이터에 대해 개인화 추천, 연관 추천한 결과물들입니다. 연관 추천에 대해 예를 들자면, 만약 사용자가 Netflix에서 드라마인 `The Last Kingdom`을 클릭했을 때, `유사한 콘텐츠`와 같은 연관 추천 카테고리를 통해 `The Vikings`, `Kingdom` 등과 같은 유사한 장르/내용의 콘텐츠들이 추천되게 됩니다. 정확하게는, 유저들이 고를 수 있는 클릭되는 형태로 동시에 여러개가 보여지는데, MAB에서는 이 콘텐츠들 하나 하나를 arm으로써 관리합니다.

- **슬롯 머신**: 콘텐츠(e.g., The Last Kingdom)
- **슬롯 머신을 시도**하는 행위: 사용자에게 콘텐츠를 보여주는 행위
- **보상**: 사용자가 콘텐츠를 선택

<br>

<br>

## Beta Distribution

MAB에서는 arm들이 보상받을 확률(클릭될 확률)들의 분포들이 각각 독립적임을 가정합니다. 단순히 생각해봐도, 모든 arm들의 클릭될 확률이 다른 것이 정상입니다. MAB에서 베타 분포를 사용하는 이유들은 다음과 같습니다.

- 보상이 1과 0(click/unclick)으로 구성되어 있다.
- 켤레 분포이다.

<br>

### 보상이 1과 0(click/unclick)으로 구성되어 있다.

**모든 서비스들에서 MAB를 click과 unclick으로만 보상을 구성하지는 않습니다. 여기서 말하는 것은 베타 분포를 사용하여 arm의 분포를 추정하는 MAB들에 국한지어 설명합니다.**

보상이 두 가지라는 것은, 사용자가 콘텐츠를 클릭하거나 안하는 행위가 베르누이 시행임을 의미합니다. 베타 분포는 alpha와 beta 값을 click/unclick에 따라 계산하여 이항 분포를 추정하는데, 계산 방법은 MAB 알고리즘 별로 다릅니다.

<br>

### 켤레 분포이다.

켤레 분포는 사전 분포와 사후 분포의 형태가 동일한 분포를 말합니다. 우리는 사전에 설정한 사전 분포인 베타 분포의 alpha와 beta 값을 사용자 피드백을 통해 갱신하고, 갱신을 통해 이항 분포이자 사후 분포인 베타 분포를 추정합니다.

<br>

<br>

## Thompson Sampling

톰슨 샘플링은 MAB에서 가장 많이 사용되는 알고리즘들 중 하나입니다. 톰슨 샘플링의 큰 특징 중 하나는, 앞서 설명한 Explore와 Exploit의 경계가 존재하지 않는다는 것인데, 이는 'beta sampling'을 사용하여 arm을 select 하기 때문입니다. 톰슨 샘플링에서는 모든 arm들에 대해 'beta sampling'이라는 것을 진행하며, 사용자 피드백(보상 정보)이 유입될 때 마다 arm들의 alpha와 beta를 업데이트합니다.

<br>

### Beta sampling

베타 샘플링은 arm들 별로 alpha와 beta 값을 사용하여 expected reward(분포의 기대값)를 계산한 후, 모든 arm들을 각각의 expected reward에 따라 확률적인 샘플링하는 행위를 말합니다. 추천 결과는 베타 샘플링을 통해 가장 클릭될 확률이 높다고 판단되는 콘텐츠들로 구성됩니다.

여기서 MAB가 왜 arm들의 분포를 수렴시키고자 하는가의 이유를 알 수 있습니다. Arm들의 분포가 수렴하지 않았을 때에는, 베타 샘플링에서의 arm의 확률이 아직 검증되지 않았습니다. Arm의 reward 확률의 분산이 클 것이며, 평균 또한 초기값에서 크게 다르지 않습니다. Arm들의 보상 확률의 분포가 수렴한다는 것은 MAB가 그 arm의 가치를 보다 정확하게 알 수 있음을 의미합니다. 즉, MAB의 Explore 행위는 arm들의 가치를 알아가는 행위이며, 베타 샘플링 시 그 가치 만큼의 확률로 arm을 select하기 위해 arm들의 분포를 수렴시키는 행위입니다.

앞에서 톰슨 샘플링에서는 Explore와 Exploit의 경계가 없다고 말씀드렸는데, 이는 위와 같은 beta sampling의 특징 때문입니다. Beta sampling에 따라 수렴되지 않은 arm일 수록 분산이 크고, 따라서 엄청 큰 reward 확률로 select 되는 경우가 많습니다. 즉, 모든 arm들이 수렴되었다면, 모든 arm들이 각자 자신의 expected reward에 따라 sampling될 것이고, 대체로 일정한 샘플링이 될 것입니다. 따라서 MAB의 Exploit과 같은 형태로 arm들을 select 하게 됩니다.

하지만 수렴되지 않은 arm들이 많을 경우, 해당 arm들의 예측 보상 확률은 매 시행마다 크게 다를 것이고, 크게 예측된 arm들이 select되어 Explore와 같은 형태를 띄게 됩니다.

<br>

### Prior-alpha/beta

위에서 설명한 베타 샘플링은 alpha와 beta의 초기값에 따라 많이 다른 양상을 띄게 됩니다. 만약 수렴된 arm들이 보통 0.02의 expected reward를 갖는다고 해 보겠습니다. 대체로 2%의 확률로 선택함을 의미합니다. 그리고, 톰슨 샘플링 논문에서 추천하는 초기값은 `alpha=1, beta=15`입니다. 이럴 경우 expected reward 값으로 0.06 정도를 갖게 되는데, 이는 수렴된 arm들에 비해 매우 높은 수치입니다. 즉, 높은 확률로 수렴된 arm 보다는 새로운 arm을 선택하게 되며, Explore가 활발하게 진행될 것입니다.

만약 `alpha=1, beta=100`으로 설정할 경우 어떻게 될까요? 이런 경우 새로운 arm의 expected reward가 수렴된 arm들 보다 훨씬 낮아질 것이고, 높은 분산으로 인해 높은 확률로 예측경우가 아니라면 대체로 수렴된 arm들을 사용하게 됩니다.

이런 것들을 생각해 봤을 때, prior-alpha/beta 값은 서비스 도메인, 데이터 성질, 그리고 서비스 목표에 따라 조절해야 함을 알 수 있습니다.

<br>

<br>

## 현실적인 고려 사항

하지만 현실에서는 사용자의 영화에 대한 견해(승률)는 항상 변합니다. 순간적인 트렌드의 변화, 새로운 콘텐츠의 등장, 요일 혹은 시간대의 변화 등에 따라 다르며, 콘텐츠를 보여줄 때(시도)의 표시되는 위치 등에 의해서도 달라집니다.

또한, 노출 대비 클릭률(Click Through Rati/o, CTR)을 생각해 보면, 시스템이 반응을 더 빨리했을 때 클릭률이 상승합니다. 그러므로 지연 시간을 늦추는 것이 CTR을 높이는 방법이지만, 빨리 반응하는데에 요구되는 시스템 리소스를 고려하여 트레이드 오프(trade off) 관계를 조율해야합니다.