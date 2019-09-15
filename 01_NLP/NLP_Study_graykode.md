# NLP Study Summary

NLP study group with graykode(김태훈), summary

<br><br>

## Day_1

- colab

  런타임 유형으로 GPU, Python 등 조작

  !ls !pip %cd $pwd(현위치)(폴더 조작은 %) 등 리눅스 명령어 조작

  session 죽으면 초기화되므로 저장 필요(docker와 유사)

  최근부터 파이토치 지원

   <br>

- pytorch

  김성훈 교수 강좌 & pytorch documents 주로 참조

  논문 구현 pytorch로, tensor는 optimizer 및 서비스 (상세한 관리)

  Keras는 너무 간략.

  pytorch.org/docs/ >> 참조(nn.Linear, nn.ConvD 등 참조)

  init과 forward만 정의해 주면, loss optimization, backward(==backpropagation)은 뒤에서 함수로 사용

   

  import Variable이 get variable 개념

  토치에는 placeholder가 없다. 절차지향처럼 node와 feed가 따로 놀지 않음.

  numpy 다루기 용이

  tensor 내부 보기 용이

  ```python
  	# y = w * x + b 구현
  		class Test(torch.nn.Module):
  			def __init__(self):
  				super(Test.self).__init__()
  				# Torch에서는 super를 꼭 써넣어야함
  				self.w = w
  				self.b = b
  				# 파라미터 정의
  			def forward(self):
  				# forward라는 함수가 무조건 있어야함
  				# 이름 틀리면 안됨
  				y = self.w * x + self.b
  				return y
  		model = Test(3,4)
  		output = model(1)
  		print(output)
  ```

 <br> <br>

## Day_2

- word2vec embedding의 차원 수(ex. 300)은 Vectorizing Model 혹은 네트워크 모델에 따라 다르다

  BERT는 512 등이 우수하다고 권장

  Fasttext는 50 100 150 등이 우수하다고 권장

- n-gram 사용 이유

  한 문장 전체에서 관련성을 찾기 힘들다.

  첫 단어와 마지막 단어의 관련성은 미미할 가능성이 높다.

- NNLM은 결과값을 다음 Input으로 넣어서, 다음 target의 n-gram 중 일부로 사용할 수 있다.

- CBOW는 target 단어를 제외한 나머지 단어들의 벡터의 평균으로 target을 맞춤

- word2vec의 단점으로, 동의어와 반의어 모두 큰 코사인 유사도를 갖는다는 점이 있다.

- charCNN은 여러 channel들로, 같은 단어조합(n-gram)을 여러 채널로 분석(feature추출의 의미)하기 위해 사용

- gpt 2

  NNLM을 사용한, 지난 주에 나온 논문

  (목적 단어 앞의 n-gram을 이용하여 목적 단어 올 조건부 확률 구하기)

  ex) '나는 오늘 소고기를 먹었다.'를 입력하면

  뒤에 많은 문장들을 생성해줌.

  <br>

- BERT에서는 Skip gram이 아닌, CBOW의 컨셉을 사용

  BERT는 단순 목표가 아닌, 모든 것에 활용 가능한 network를 만드는게 목표

  (classification, seq2seq 등)

  <br>

- Code Review

  n_step == n-gram 지정 변수

  n_hidden == paper에서 변수 H값

  m = 임베딩 차원 수

  X = self.C(X) :: 임베딩에서 X에 해당하는 행을 가져옴

  <br>

- Fasttext

  !git clone https://github.com/e9t/nsmc

  한국어 패텍

  __label__1 안녕하세요 저는 조동빈 입니다.

  __label__ 을 앞에 넣어야 함(input format)

  fasttext document로 파라미터 참조

  __label__0

  __label__1

  :: 0은 부정, 1은 긍정으로 classification할 때의 target을 표시

  네이버 평점 긍정 부정

  띄어쓰기로 tokenizing

- attention :: cs224, 조경현 교수 강의, rat'so 블로그(attention mechanism)

  <br>

- one-shot learning

  :: 영어 한국어가 혼용되어 있는 input을 한번에 일본어로 번역해주기

- 파파고에 '나는 너를 love해'라고 일본어 본역기를 돌리면,

  '와타시와 오마에오 love데스' 등으로 섞여도 번역해줌

- 구글 sentencepiece

 <br> <br>

## Day_3

RNN's LSTM

GRU

Seq2Seq Encoder-Decoder

BLEU

- SimpleRNN

  RNN cell은 모두 재귀식이다 (펼쳐져 보이나, 모두 하나의 셀에서 이루어지나 보기 쉬우라고 펼쳐놓음)

  RNN cell의 output은 모든 hidden 값들의 step별 값들을 저장한 값

  num_direction은 uni 인지 bi인지 결정 (1 or 2)

  output은 마지막 layer의 값만 저장된다

  layer는 최대 3~4개 층만 쌓는다. 더 쌓을 시 explosion, vanishing 발생 (역전파가 어렵다. tanh 하이퍼볼릭탄젠트 -1~1을 사용하기 떄문에. 분수로 이루어져있기 떄문에 미분 시 어쩌구저쩌구

- LSTM

  LSTM long short term memory

  cell state = long term memory(쭉 돌면서 과거 정보까지 계속 간직)

  hidden state = short term memory (전, 이번 정보를 다룸)

  output은 hidden state만을 포함(심플 Rnn은 hidden state만 있다)

- Teacher/non-Teacher forcing

  Teacher forcing이 일반적으로 우리가 알던 방식

  input sequence를 넣고, 우리의 output을 target과 비교하여 loss 게산

   

  non-teacher forcing은 전 step의 output이 다음 step의 input으로 들어감

  (loss 계산은 동일하게 target과 output 비교)

  계산 속도가 느려지며, inference에서 문제 발생

- CNN은 '조대협'님이 bcho.tistory.com 참조

  TextCNN에서 단어의 벡터열들을 필터로 묶고,

  채널을 분리(채널을 분리하는 W도 학습되야 하는 파라미터)하여 학습하고, 원래 형태로 프로젝션

  그림마다 1Xn(vector 차원수)의 feature map(skip gram 결과물)을 모두 max pooling 후 concat 하여 projection

- Seq2Seq 모델은 pretrain 시킨 word ebbeding을 사용할 수도, 자체적으로 임베딩을 만들수도 있다.

- Attention

  I go to school

  I to school go 두 문장의 차이를 인지할 수 있도록 도움

  (love가 '사랑'과 매칭되도록)

  encoder의 output을 그냥 seq2seq에서는 아예 무쓸모

  attention을 적용한 seq2seq에서 encoder input과의 연결성을 찾기 위해 encoder output vector를 사용하여 decoder output을 생성

  매 decoder의 timestep마다 encoder 모든 timestep의 h 값과 비교

  eij 계산 시 그냥 si-1과 hj를 dot product 해도 scalar값이 되기 때문에 이를 eij로 사용해도 됨.

  Contextual matrix 중에서 현재의 query(디코더 위치)가 어느 곳에 가장 큰 영향을 받는가를 찾는 것

  인코더의 output(contextual matrix)과 어텐션 벡터의 소프트맥스화(알파i)를 곱해줌 >> F알파

  디코더 아웃풋(디코더의 타임스텝별 값을 보유)과 어텐션벡터를 컨캣 하여 이를 최소화하는 학습을 진행

- 구현

  non-teacher forcing처럼 timestep을 돌면서 cell을 step마다 생성해줘야함 ( query를 contextual matrix와 비교해줌)

  attention score(scalar값인 e)

- Self attention

  I like apple, and I like it.

  it이 무엇을 가르키는지

- Transformer

  encoder+decoder 존재

  BERT는 transformer의 encoder만

  gpt는 transformer의 decoder만 사용 

- BERT 학습 방식

  1. 문장 중 10~15 퍼센트를 가려서 빵꾸뚫기
  2. 여러 문장들 중, next 문장이 아닌, 엉뚱한 문장을 바로 뒤에 배치시키기

  CBOW 방식과 유사하며, 과정에 concatenate 사용

  (window의 각 단어와 유추 단어와의 계산들을 모두 concatenate)

  (sum을 하면 정보가 손실됨 :: concatenate 하면 보완) 

  k =1 grid search :: gridy하게

  k = 4 beam search :: tree 구조로