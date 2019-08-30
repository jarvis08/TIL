# Variable

## 변수, Variable이란

### TensorFlow 공식 홈페이지 설명

> A TensorFlow **variable** is the best way to represent shared, persistent state manipulated by your program.
>
> Variables are manipulated via the `tf.Variable` class. A `tf.Variable` represents a tensor whose value can be changed by running ops on it. Specific ops allow you to read and modify the values of this tensor. Higher level libraries like `tf.keras` use `tf.Variable` to store model parameters.

> TensorFlow **변수**는 프로그램에 의해 변화하는 공유된 지속 상태를 표현하는 가장 좋은 방법이다.
>
> 변수는 [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) 클래스에서 처리된다. 하나의 [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable)은 하나의 텐서를 표현하는데, 텐서값은 텐서에 연산을 수행하여 변경시킬 수 있다. 특정한 연산은 이 텐서값을 읽고 수정한다. [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras) 같은 좀 더 고수준의 라이브러리는 모델 파라미터를 저장하는데 [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable)을 사용한다.



### tf.Varible을 사용하는 이유

`tf.Variable`은 그 이름으로 부터 알 수 있듯이, 변수를 저장할 수 있는 객체입니다. 그런데 왜 일반적인 파이썬 코딩처럼 `w = 0`과 같은 형태로 변수를 선언하면 안되는 것일까요? 그 이유는 텐서플로(TensorFlow)가 모델을 학습할 때 텐서(Tensor)와 그래프(Graph)를 활용하기 때문입니다. 그리고 우리는 우리가 선언 할 변수들을 텐서 형태로 다루기 위해 `Variable` 객체를 사용해야 합니다. `tf.Variable`을 사용하여 선언한 변수는 사라지지 않고 지속되며, 공유됩니다. 사용자가 작성한 연산의 묶음(프로그램)은 작성된 연산을 수행하여 `tf.Variable`의 값을 갱신해 나아갑니다.



## Variable 다루기 

### 변수 의 생성 및 할당

`tf.Variable` 객체를 생성하기 위해서는 초기값을 선언해 주어야 하며, 그 형태는 다음과 같습니다.

`변수명 = tf.Variable(초기값)` 

아래의 코드 블럭은 텐서플로 공식 홈페이지에 작성된 코드 예시입니다.

```python
import tensorflow as tf

my_variable = tf.Variable(tf.zeros([1., 2., 3.]))
```

위 코드는 `my_variable`이라는 변수에 `tf.Variable` 객체를 저장합니다. `tf.Variable` 객체는 shape이 [1, 2, 3]이며, 모든 값이 0으로 이루어진 데이터의 텐서입니다. 데이터 타입(`dtype`, data type)은 따로 지정되지 않았으므로 `1.` 이라는 실수의 값을 참고하여 자동으로 `tf.float32`라는 실수를 의미하는 객체로 저장됩니다.

- 특정 장치를 지정하여 생성 및 저장하기

  `tf.device(할당 위치)`를 통해 원하는 장치에 변수를 저장할 수 있습니다. 지정하지 않는다면 자동으로 가장 빠른 장치에 저장되며, GPU가 가용하다면 대부분의 변수들이 자동으로 GPU에 저장됩니다.

  ```python
  import tensorflow as tf
  
  with tf.device('/device:GPU:1'):
      v = tf.Variable(tf.zeros([10, 10]))
  ```

  참고. `tf.distribute` API를 사용하여 다양한 분산 설정을 하는 것이 이상적인 장치 할당 방법입니다.

  

### 변수 조작하기

1. `tf.Tensor`처럼 사용하는 방법

   ```python
   import tensorflow as tf
   
   v = tf.Variable(0.0)
   w = v + 1
   tf.print(w)
   ```

   결과

   ```
   1
   ```

2. 메서드를 활용하는 방법

   ```python
   import tensorflow as tf
   
   v = tf.Variable(0.0)
   v.assign_add(2)
   
   tf.print(v)
   
   t = v.read_value()
   print(t)
   ```

   결과

   ```
   <tf.Variable 'UnreadVariable' shape=() dtype=float32, numpy=2.0>
   2
   tf.Tensor(2.0, shape=(), dtype=float32)
   ```

   - 대부분의 텐서플로 옵티마이저(optimizer)는 특정 알고리즘에 대해 변수값을 효율적으로 업데이트 할 수 있는 연산을 보유하고, 사용하고 있습니다.

   

### 사용 중인 변수 목록 확인하기

`tf.Module` 클래스(Class)를 상속받으면 `variables` 혹은 `trainable_variables` 메서드를 사용하여 사용중인 모든 변수들을 return 받음으로서 추적(track)할 수 있습니다.

```python
import tensorflow as tf

class MyModule(tf.Module):
  def __init__(self):
    self.v0 = tf.Variable(1.0)
    self.vs = [tf.Variable(x) for x in range(10)]
    
class AnotherModule(tf.Module):
  def __init__(self):
    self.m = MyModule()
    self.v = tf.Variable(10.0)
    
my_var = AnotherModule()
print(len(my_var.variables))
```

결과

```
12
```

`MyModule` 클래스에는 11개(1 + 10)의 데이터가 존재하며, `AnotherModule` 클래스에는 `MyModule` 클래스로 부터 받은 11개의 데이터와 자기 자신의 1개의 데이터를 합한 총 12개의 데이터가 존재합니다.

레이어(layer)를 형성하고 있다면, `tf.Module` 대신 `tf.keras.Layer`를 상속받는 것이 더 좋은 선택일 수 있습니다. 케라스 인터페이스를 형성함으로서 케라스에 완전하게 통합될 수 있으며, 이로 인해 `model.fit`을 포함하여 잘 통합된 다른 API들을 사용할 수 있습니다. `tf.keras.Layer`의 변수 추적 방법은 `tf.Module`의 방법과 동일합니다.