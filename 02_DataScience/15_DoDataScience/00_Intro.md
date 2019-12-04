# Intro

참고자료: [Do Data Science](https://book.naver.com/bookdb/book_detail.nhn?bid=7363405), _책을 공부하면서 필요한 내용을 발췌, 추가 설명, 정리했습니다._

<br>

### Sampling Distribution

Sampling distribution is the uncertainty created by sampling process. If we extract set of emails from larger population, it will be just a single realization, can't represent all of the population.

e.g., 같은 과정으로 샘플을 재추출 하더라도, 이전과는 다른 관찰 결과를 얻는 경우가 그 예입니다.

<br>

### Definition of Big Data

1. "Big" is a moving target.

   '1 petabyte 이상의 크기'와 같은 정의는 의미가 없습니다. 현재 사용되는 하드웨어가 감당할 수 있는 여부에 따라 그 사이즈가 달라지며, 과거에는 이보다 훨씬 작은 크기를 의미했을 것입니다.

2. "Big" is when you can't fit it on one machine.

3. Big Data is a cultural phenomenon.

4. The 4 Vs

   1. Volume
   2. Variety
   3. Velocity
   4. Value

<br>

<br>

## The Rise of Big Data

In Cukier and Mayer-Schoenberger article, they argue that the Big Data revolution consists of 3 things:

- Collecting and using a lot of data rather than small samples
- Accepting messiness in your data
- Giving up on knowing the causes

데이터의 크기가 매우 방대하기 때문에 이해할 필요가 없으며, 'keeping track of the truth'를 의미하므로 sampling error에 대해 걱정할 필요가 없습니다.

<br>

### N=ALL

모든 데이터를 사용한다고 해서 전부를 아는 것은 아닙니다. 투표 데이터를 모은다고 가정했을 때,

1. 모든 사람들이 투표를 하지는 않는다.

   투잡을 뛰는 사람은 바빠서 투표할 시간이 없습니다.

2. 모든 투표가 유의미하지 않다.

   투표를 했다고 해도, 무효 표를 작성했을 수 있습니다.

<br>

### Data is not objective

데이터는 객관적인 의미를 내포하지만은 않습니다. 같은 능력과 표면적 자격을 갖춘 남자와 여자가 있으며, 두 사람이 같은 회사에서 일하고 있다고 가정해 보겠습니다.

1. 여자는 남자보다 언제나 일찍 퇴근합니다.
2. 여자는 남자보다 승진이 늦습니다.

그런데, 회사 내부적으로 여성 직원에 대한 처우가 남자 직원에 대한 처우보다 좋지 않습니다. 하지만 이러한 내용은 데이터에 포함되어 있지 못합니다. 만약 우리의 모델이 신입 사원을 채용할 때 사용된다고 한다면, 모델은 동일 스펙의 여성에 비해 남성을 더 채용하도록 할 것입니다.

<br>

### n=1

사실 n=1인 경우는 과거에나 지금이나 사용되지 않는 샘플 개수입니다. 그런데 과거와 현재의 차이는 n=1일 지언정, 그 하나의 샘플에 많은 데이터들이 포함된다는 것입니다. 이름, 주소, 전화번호와 같은 것 뿐만 아니라 사소한 모든 것들이 데이터로 존재합니다. User-level Modeling이 이러한 사실에 기반하여 만들어질 수 있는 모델입니다.

<br>

<br>

## Modeling

모델링은 우리가 모으는 데이터로부터 모델을 만드는 작업을 말합니다. Model이라는 단어는 다양하게 사용됩니다.

- Data Model

  Choosing to store one's data, which is the realm of databse managers.

- Statistical Model, Mathematical Model

<br>

### Definition of Model

Statisticians and data scientists capture the uncertainty and randomness of data-generating processes with mathematical functions that express the shape and structure of the data itself.

A model is our attempt to understand and represent the nature of reality through a particular lens, be it architectural, biological, or mathematical.

A model is an artificial construction where all extraneous detail has been removed or abstracted. Attention must always be paid to these abstracted details after a model has been analyzed to see what might have been overlooked.

In the case of a statistical model, we may have mistakenly excluded key variables, included irrelevant ones, or assumed a mathematical structure divorced from reality.

<br>

### Statistical Modeling

작업을 수행할 때, 데이터와 코드를 작성하는 것에 앞서, 모델을 통해 밑바탕이 되는 전체 프로세스의 큰 그림을 그려보는 것이 유용합니다. 그림은 무엇이 먼저 오며, 무엇이 어떤 영향을 미치는지, 무엇이 어떤 일을 야기하는지 등의 고민을 말합니다.

그림을 그리는 방법 또한 다양합니다. 어떤 사람들은 일반화를 위해 수학적 표현을 사용하기도 하지만, 수식에 사용되는 파라미터들은 알려져 있지 않기 때문에 어렵습니다.

e.g., $y = \beta_{0} + \beta_{1}x$

- Greek letters: Parameters
- Latin letters: Data

또 어떤 사람들은 data flow의 diagram을 그리는 등의 작업을 수행하기도 합니다.

<br>

### Practicing building models

Start simply and then build in complexity.

1. Plot histograms
2. Look at scaterplots to start getting a feel for the data
3. Try writing sth down

위 작업은 똑똑한 방법이라고 말할 수는 없지만, 모델을 제작하는 것에 좋은 공부가 됩니다. 위 연습 과정을 수행했으며, Linear function과 같이 어떤 모델을 작성했다고 해 봅시다. 그다음에는 다음 두 가지를 반복하며 모델을 개선합니다.

- 이 모델이 정말 나의 데이터에 적절한 모델인가?
- 어떻게 하면 보다 의미 있는 모델로 개선시킬 수 있는가?

모델의 복잡성과 성능은 trade-off의 관계를 가집니다. 제작하기 쉽고 이해하기 쉬운 모델은 좋은 성능을 가졌다 말할 수 없을 수도 있습니다. 그런데, 몇 시간에 걸쳐 제작한 90%의 정확도를 가진 모델과, 한달에 걸쳐 제작한 92% 정확도의 모델이 있습니다. 어떤 모델이 더 유용한 것인지는 상황에 따라 다를 것입니다.

