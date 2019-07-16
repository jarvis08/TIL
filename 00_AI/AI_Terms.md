# AI_Terms

---

- **Epoch, Iteration, Batch**
  
     - `Epoch` : 전체 데이터셋을 훑는 횟수
     - `Batch` : 전체 데이터셋 중 몇개의 데이터씩 훑을 것인지
     - `Iteration` : step횟수(백프로퍼게이션해서 수정하는것까지가 1step)
     
     i.g., 전체 2000개 데이터 일 때, 500의 batch와 4의 iteration(step)이면, 1 epoch.
     
- **Kernel**
  
     : 컴퓨터 운영체계의 가장 중요한 핵심으로써 운영체계의 다른 모든 부분에 여러 가지 기본적인 서비스를 제공한다. 커널은 셸(shell)과 대비될 수 있는데, 셸은 운영체계의 가장 바깥부분에 위치하고 있으면서, 사용자 명령어에 대한 처리를 담당한다. 일반적으로 커널에는 종료된 입출력연산 등 커널의 서비스를 경쟁적으로 요구하는 모든 요청들을 처리하는 [인터럽트](https://terms.naver.com/entry.nhn?docId=782406&ref=y) 처리기와 어떤 프로그램들이 어떤 순서로 커널의 처리시간을 공유할 것인지를 결정하는 스케줄러, 그리고 스케줄이 끝나면 실제로 각 프로세스들에게 컴퓨터의 사용권을 부여하는 수퍼바이저(supervisor) 등이 포함되어 있다.
      또한 커널은 메모리나 저장장치 내에서 운영체계의 주소공간을 관리하고, 이들을 모든 주변장치들과 커널의 서비스들을 사용하는 다른 사용자들에게 고루 나누어주는 메모리관리자를 가지고 있다. 커널의 서비스는 운영체계의 다른 부분이나, 흔히 시스템 호출이라고 알려진 일련의 프로그램 [인터페이스](https://terms.naver.com/entry.nhn?docId=782407&ref=y)들을 통해 요청된다.
<https://terms.naver.com/entry.nhn?docId=782920&cid=42111&categoryId=42111>
     
- Random initialization
     세타값이 0이면 모두 동일한 값의 hidden layer들로 생성
     <https://blog.naver.com/dreamclouud/221339382795>

- **Text normalization**
         공백제거, 구두점 삭제, 대문자/소문자 변환 등

- **Batch normalization**
  
     - 목적
     
       activation에 들어가는 input range를 제한시킴으로써 Internal covariate shift 제거
       input X가 weight와 곱해지고, activation을 거쳐 hidden layer를 갈 때 activation 이전에 scaling을 하여 weight에 의한 변화량을 감소시키는 것. weight가 너무 훽훽 변하면 학습이 제대로 되지 않고 산으로 갈 수 있음.
     
     - Internal covariation shift
       학습하면서 hidden layer들이 각각 마음대로 weight값 설정되어 이상한 output 받는 오류
     
     - Batch Normalization 효과
     
       1. 빠른 학습(learning rate를 상승 가능)
     
       2. initialization 고려 필요성 감소
     
       3. regularization effect, mini-batch의 분산과 평균에 따른 변화가 있기 때문에 Dropout의 필요성 감소
     
- **Ground truth, Wikipedia**
     지도학습의 classification의 training sets' accuracy
     DB의 형상/속성 등에 대하여 정확성과 완성도를 검증할 때 현장검증을 대신할 수 있는 정확성과 완성도의 참조 자료
     term used in statistics and machine learning that means checking the results of machine learning for accuracy against the real world.
     
     In machine learning, the term "ground truth" refers to the accuracy of the training set's classification  for supervised learning techniques. This is used in statistical models to prove or disprove research hypotheses. The term "ground truthing" refers to the process of gathering the proper objective (provable) data for this test. Compare with gold standard.
     Bayesian spam filtering is a common example of supervised learning. In this system, the      algorithm is manually taught the differences between spam and non-spam. This depends on the ground truth of the messages used to train the algorithm – inaccuracies in the ground truth will correlate to inaccuracies in the resulting spam/non-spam verdicts.
     
- **Teacher-forcing**

     - 학습 효율적으로 하기 위해 전 output을 이번 input으로 사용
     - Teacher forcing is a method for quickly and efficiently training recurrent neural network models that use the output from a prior time step as input.
     - It is a network training method critical to the development of deep learning language models used in machine translation, text summarization, and image captioning, among many other applications.
       <https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/> 

- **Highway Network**
         확률적인 계산 및 통계의 normalizaions 함으로써 더 깊게, 그리고 layer들을 skip
         Highway Networks, adapts the idea of having “shortcut” gates, where it can circumvent certain layers of propagation of information to go “deeper”, in terms of skipping layers.
         This means, that, under certain mathematical probabilities and normalizations of statistics, it can simply, go deeper a number of layers, by activating the gated function. Like a power-up! Wooh!
         This, has the effect of skipping some stages of where a large descent is needed, as in, just similarities are abound in such a leap that we can just, “overstep” that co-relativity and go to “more useful areas”.
         <https://www.quora.com/What-are-the-differences-between-Highway-Networks-and-Deep-Residual-Learning> 

- **Residual learning**
         Deep Residual Learning, is the idea of having residual remnants of information scattered across certain layers, to account for “bread crumbs” of learning, as it goes deeper and deeper into the cannyon of information propagation.
         This, is where the residual part propagation comes in.
         At times, this means that, to account for actuallity of hyperplane dimension calculations in terms of our residual vectors, is that we must “jump” a dimension, now and then, to account for better interplay of dimensions.
         <https://www.quora.com/What-are-the-differences-between-Highway-Networks-and-Deep-Residual-Learning>

- **Activation function**
  
     - 기능 : 출력값의 활성화를 일으키게 하는 함수
     
     - 목적 : Data를 비선형으로 변형시키기 위함
       (선형일 시 DNN의 구조가 무의미하며, 하나의 은닉층이면 충분) 

- **Deep Layer**의 장점
  
     1. 같은 수준의 정확도일 때 매개변수가 더 적게 필요
     2. 필요 연산의 수를 감소시킴
     
- **Placeholder**

     - 실제 학습에 사용할 데이터를 저장하며, 학습에 사용

     - 초기값을 제공하지 않으며, session의 feed_dict를 통해 데이터 feeded

     - rank라 부르는 차원의 단위를 사용

       - `rank0` : scalar, tf.placeholder(tf.float32, (), name='somerank0")

       - `rank1` : vector, tf.placeholder(tf.float32, (None))

       - `rank2` : matrix, tf.placeholder(tf.float32, (None, 124))  

         <https://blog.naver.com/acelhj/undefined>

         ```python
         # 2 rank이며 4개의 필드를 가지는, 전체 크기를 알지 못하는 tensor인 'input'
         input = tf.placeholder(tf.int32, shape=[None,4], name="input")
         ```

- **Variable**
  - 모델의 W,b와 같은 학습 가능한 변수에 사용
  - 학습의 결과로서, 학습이 되는 대상
  - 초기값 제공 요구
    <https://blog.naver.com/acelhj/undefined> 

- **variable_scope**
         변수들의 세트
         <https://blog.naver.com/acelhj/undefined> 

- **CRF, Conditional Random Fields**, 조건부 무작 위장

   CRF는 HMM과 근본적으로 다르지는 않습니다. HMM은 아주 단순히 말하자면 현재 상태에서 다음 상태로 전이 확률과 특징 확률을 곱하는 방식이지요. 아주 거칠게 말해서, CRF는 상태 함수가 여럿으로 구성된 HMM으로 말할 수 있습니다. (물론 이 말을 그대로 믿으면 안됩니다. 단순한 묘사일 뿐입니다.) 

  **HMM** 은 워낙 단순한 모델이기 때문에 **전이함수 \* 상태함수**만 생각합니다. 여러 함수들을 섞으려면 곱하기만 하면 0에 가까운 작은 값을 나타내는 함수의 영향이 너무 크고, 더하기만 하면 1에 가까운 큰 값을 나타내는 함수의 영향이 너무 큽니다. 이 **문제는 아무리 가중치를 주어도 해결하기 힘듭**니다. 그래서 참 많이 고민이 됩니다.

  **CRF는 여러 특징 함수를 섞는 문제를 해결하는 모델**입니다. **전이함수와 특징함수를 가중치를 곱한 채로 몽땅 더해서 지수값을 취한 다음에 정규화**를 합니다. 나중에는 특징함수와 전이함수의 구분조차 없애버리더군요. -_-;;; 해당 수식은 http://www.inference.phy.cam.ac.uk/hmw26/papers/crf_intro.pdf 의 4페이지에 모두 나와있습니다. 이게 워낙에 튼튼한 방법이라서, 각 함수가 1~0 사이에서 어떠한 자유로운 결과를 내도 됩니다. 나중에 보정을 하면 되니까요. 이게 잘 이해가 안되면 극단적으로, 대충 함수를 열거하면 알아서 조정해주는 프로그램으로 생각하면 됩니다.

  특징 함수의 종류에는 '이 단어의 첫 글자가 대문자이면 명사일 확률이 1이다' '앞 단어가 the 이면 명사일 확률이 1이다' 등의 단순한 함수조차도 들어갈 수 있습니다. 물론 더 복잡한 함수들도 얼마든지 들어갈 수 있습니다. **각 특징 함수들은 가중치를 가집**니다. **학습 데이터를 통해서 그 최적값을 찾아내야 좋은 결과**를 얻을 수 있습니다. 이 학습 과정에 관한 연구도 꽤 있습니다. 물론 이를 자동으로 하는 오픈소스 패키지도 있습니다.

  HMM은 매우 단순한 모델링 방법이며, forward-backward와 Viterbi 같은 단순한 해석 알고리즘이 주 해석 방법으로 쓰입니다. **CRF도 Viterbi와 같은 해석 알고리즘을 활용할 수 있**습니다. 이러한 알고리즘들은 **통신에서 잡음 제거를 하는 분야에도 널리 쓰이**니, 서로 참고하면 좋습니다.

   CRF 를 연쇄로 적용할 수도 있습니다. **Higher-order CRF**는 **임의로 정한 길이만큼 연쇄**하는 겁니다. bigram HMM을 한 번 더 묶어서 trigram HMM을 만드는 식이랄까요? 당연히 **연쇄를 길게 할 수록 품질은 좋아**집니다. **하지만 한 단계를 넘어갈 때마다 지수급수적으로 계산량과 필요한 학습 자료량이 늘어나므로, 5단계 이상은 힘듭**니다. 그래서 나온게 **semi-CRF**입니다. 이건 **품질이 좋은 긴 연쇄만 남기고, 나머지는 짧은 연쇄로 인식하는 가변 연쇄 방식**입니다. 속도도, 품질도 좋다고 합니다.

  <http://ysw1209.blogspot.com/2010/02/crf-conditional-random-field.html> 

- **GRU, Gated Recurrent Units**

  LSTM의 변형이며, 다음 논문을 통해 처음으로 제시된 RNN cell의 한 종류

  - Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation

    <https://arxiv.org/pdf/1406.1078.pdf>