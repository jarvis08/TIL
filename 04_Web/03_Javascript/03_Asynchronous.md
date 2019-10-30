# Asynchronous & Non-blocking

**Python**의 경우 **Synchronous**하며, **Blocking**되어 있기 때문에 `sleep()`을 넣고 싶은 곳에 넣으면 됩니다. 따라서 다음과 같은 것이 가능합니다.

```python
print('start')
sleep(5)
print('end')
```

하지만 다음과 같은 JS의 sleep은 적용되지 않습니다. 이를 확인해 보기 위해 `setTimeout()`을 사용해 보겠습니다.

`setTimeout(sleep 이후 실행하는 함수, sleep시간)`

`sleep 이후 실행하는 함수`에 함수를 지정해야 sleep 시간을 기다립니다. 만약 sleep(단위:`ms`) 시간 이후의 행위를 `setTimeout()` 바깥에 선언해 둔다면, 이는 적용되지 않습니다.

```javascript
const nothing = () => {}

console.log('start')
setTimeout(nothing, 3000)
console.log('end')
```

위와 같은 JS 코드는 start, end가 바로 출력된 뒤 3초 후에 실행이 종료됩니다. Python과 같은 결과를 받고자 한다면 다음과 같이 사용해야 합니다.

```javascript
const nothing = () => console.log('end')

console.log('start')
setTimeout(nothing, 3000)
```

이러한 성질을 **Asynchronous**하며, **Non-blocking**하다고 합니다. 이는 Browser를 조작하는 데에는 매우 중요한 성질입니다. 만약 Python처럼 Synchronous하게 Blocking한 상태로 sleep이 진행된다면, **그 시간 동안 사용자는 브라우저를 이용할 수 없게 됩니다**.

따라서 Javascript는 다음 두 가지 작업에 대해서는 Asynchronous하게 작업합니다.

1. **종료를 예측할 수 없는 작업**

   i.g., `XHR, XMLHTTPResponse`: 언제 요청에 대한 응답이 돌아올지 알 수 없다.

2. **시간이 오래걸리는 작업**

이러한 작업들은 **Call Stack과 Queue**에 쌓아둔 후, **Event Loop**로 작업 완료를 알립니다. 이름이 Call **Stack**이기 때문에 async 함수끼리는 순서가 있을 것 같지만, 그렇지는 않습니다. Asynchronous 작업들이 여러개 있을 경우에도 시작 순서와 상관 없이, 작업이 완료되는 대로 끝나게 됩니다. 만약 작업 시작 순서대로 끝내고 싶을 경우 Call Back을 이용하여 async 함수 안에 async 함수를 넣어야 합니다. Call Back 안에 Call Back을 넣는 과정이 계속 반복되어야 하며, **Call Back Hell**이 펼쳐지기도 합니다.

물론 **JavaScript 자체는 Synchronous** 합니다. **하지만, 일부 함수는 Asynchronous** 합니다. 그 중 대표적인 예시가 `setTimeout()`입니다.

**`addEventListener(event, 동작함수)`** 또한 **async하게 대기**하던 중, **`event`가 유입됐을 때에만  `동작함수`를 수행**하게 됩니다. 따라서 코드 상으로 `addEventListener()` 보다 이후에 선언된 함수를 `동작함수`에서 사용하더라도 아무런 에러가 발생하지 않습니다. `event` 발생 시 `addEventListener()`의 `동작 함수`가 **Call Back**되어 수행됩니다.

이러한 JS의 특징으로 인한 **또 다른 장점은 Multi-threading**입니다. JS 언어 자체적으로 멀티쓰레딩이 너무나 활발히 이루어지기 때문에 **많은 사용자들의 요청을 처리해야 하는 서버 언어로 사용하는 것에 있어서 매우 우수한 성능**을 보이며, 많이 사용되고 있습니다.

<br>

### Async가 유용한 상황의 예시

만약 웹 페이지에 고해상도 이미지를 표시해야 하며, 이미지 위에 어떤 텍스트를 띄워주고자 한다. 또한 이미지가 모두 load된 이후에 텍스트를 띄워야만 한다고 가정했을 때, 이는 Async 한 상황을 Sync하게 처리해야만 합니다. 그리고 이를 구현하기 위해 Async 함수를 통해 그림을 load하며, Call Back을 통해 텍스트를 그림 위에 표시합니다.

