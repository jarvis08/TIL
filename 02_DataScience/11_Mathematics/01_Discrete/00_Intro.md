# Intro

**이산**: 낱낱의 개체가 떨어져 있다는 말로, '연속'과 대칭되는 말

**이산수학**: 이산적인 수학적 구조를 다루는 수학이며, 이산적인 대상과 이산적인 방법을 사용하는 수학

**조합론**: 이산적 구조에 대한 존재성, 헤어리기, 최적화 문제, 구조분석 등을 다루는 수학의 분야

<br>

### 합의 법칙과 곱의 법칙

배열의 개수를 헤아리는 가장 기본이 되는 법칙

- **합의 법칙**

  **동시에는 일어나지 않는 두 사건 X, Y**에 대하여, X가 일어나는 경우가 m가지, Y가 일어나는 경우가 n가지 일 때, X 또는 Y 중 어느 하나가 일어나는 경우는 m+n 가지이다.

- **곱의 법칙**

  사건 X가 일어나는 방법이 m가지이고, 그 **각각의 m가지 경우에 사건 Y가 일어나는 방법이 n가지**라고 한다면, X와 Y 모두가 일어나는 방법은 mn가지이다.

<br>

<br>

## 집합, 관계, 함수

### 집합

**멱집합**: 집합 X의 부분집합 전체의 집합(공집합 포함)이며, 원소의 개수는 `2**|X|`와 같다. `|X|`는 집합 X의 원소의 개수이다.

- **분할**

  집합 X를 교집합이 존재하지 않는 2개 이상의 집합으로 나눌 때, 이 집합들을 분할이라고 한다.

  - 집합 X의 분할인 집합들이 있을 때, 모든 분할들의 원소의 개수들의 합(`|분할_1|` + `|분할_2|` + ..)은 |X|와 같다.

<br>

### 관계

**데카르트 곱**: 집합 A, B에 대하여 `A와 B의 데카르트 곱 = A X B = {(a, b)| a와 b는 각각 A와 B의 원소}`인 **순서쌍 전체의 집합**

```
예시)
A = {1, 2}, B = {p, q, r} 일 때,
A X B = {(1, p), (1, q), (1, r), (2, p), (2, q), (2, r)}
```

A X B의 임의의 부분집합 R을 A, B 사이의 **관계**라고 한다. (a, b)가 R의 원소일 때 `aRb`와 같이 나타날 수 있으며, a는 R에 의하여 b와 관계지어져 있다고 한다. 특히 A = B 일 때 R을 **A상의 관계**라고 한다.

- A, B 사이의 관계 R의 원소들의 첫 번째 성분 전체의 집합을 R의 **정의역**이라 하며, 두 번째 성분 전체의 집합을 R의 **치역**이라고 한다.

<br>

### 함수

다음의 관계 조건을 만족하는 f를 **함수**라고 한다.

1. f의 정의역이 A이다.
2. A의 원소 각 a에 대하여, (a, b)가 f의 부분집합임을 만족하는 B의 원소 b가 오직 하나 존재한다.

여기서의 b를 a의 **함수값**이라고 하며, f(a)로 표현한다. 또한 A에서 B로 가는 함수는 `f: A -> B`라고 표현한다.

