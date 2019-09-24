# Overall of Sorting Algorithms

**정리 요점**

1. **정렬 방법**
2. **Stable**/**Unstable**
3. **최선**의 경우 시간복잡도
4. **어떤 경우에 최선**의 시간 복잡도를 갖는가
5. **평균**적인 시간 복잡도
6. **최악**의 경우 시간복잡도
7. **어떠한 경우에 최악**의 시간 복잡도를 갖는가

![img](https://github.com/jarvis08/TIL/raw/master/02_ComputerScience/02_Algorithm/assets/Sort_Complexity.png)

# O(n)

## 삽입 정렬, Insertion Sort

### 1. 맨 앞의 요소부터 시작하며, 배열의 맨 앞 부분에 정렬된 형태로 끼워 넣는다. 과정 중에는 정렬된 부분과 정렬되지 않은 부분으로 나뉜다.

자료가 소량인 경우에 유리하며, 알고리즘이 간단하다는 장점이 있다.

하지만 이미 정렬되어 있는 자료구조에 자료를 하나씩 삽입/제거하는 경우, 오버헤드가 매우 적다.

<br>

### 2. Stable

<br>

### 3. 최선 - O(`n`)

<br>

### 4. 이미 정렬되어 있는 경우로, 비교만 하면 된다.

<br>

### 5. 평균 - O(`n^2`)

<br>

### 6. 최악 - O(`n^2`)

<br>

### 7. 역순으로 정렬된 상태일 때

<br><br>

## 버블 정렬, Bubble Sort

### 1. 좌측에서 우측으로 인접한 두 요소를 반복적으로 비교하며 크기가 큰 요소를 계속해서 뒤 인덱스로 옮기며 정렬한다.

오름차순으로 정렬하고자 할 경우 한 싸이클 진행 시 가장 큰 값이 맨 뒤에 저장된다. 그리고 싸이클이 진행될 수록 싸이클 당 진행 작업 회수는 하나씩 줄어든다(`전체 배열의 크기` – `현재까지 순환한 바퀴 수`).

<br>

### 2. Stable

시간복잡도는 거의 모든 상황에서 `최악`의 성능을 보여준다.

`공간복잡도`는 단 하나의 배열에서 진행하므로 `O(n)`이다.

<br>

### 3. 최선 - O(`n`)

<br>

### 4. 이미 정렬된 자료

<br>

### 5. 평균 - O(`n^2`)

<br>

### 6. 최악 - O(`n^2`)

<br>

### 7. 모든 요소들을 비교할 때

<br><br>

## 기수 정렬, Radix Sort

### 1. 다중키를 사용하며, 자리수가 같은 요소들끼리 비교하여 정렬하는 방법

- LSD: 가장 작은 자리수부터 비교하는 방법
- MSD: 가장 큰 자리수부터 비교하는 방법

다중키를 사용하여 비교적 많은 공간을 사용하지만, 상대적으로 수행시간이 적게 든다.

비교 연산을 하지 않으며 정렬 속도가 빠르지만 데이터 전체 크기에 기수 테이블의 크기만한 메모리가 더 필요하다.

자리수를 비교해서 정렬하는 방식이므로, 자리수가 없는 것들은 정렬할 수 없다.

(i.g., 부동소수점 형태의 자료형)

<br>

### 2. Stable

<br>

### 3. 최선/평균/최악 - O(`n`)

<br>

### 4. 다중키를 사용하여 비교적 많은 공간을 사용하며, '자리수'가 존재하는 자료형만 가능(부동소주점 형태는 불가)

<br><br>

## 계수 정렬, Counting Sort

### 1. 요소 별 개수를 세어 카운팅 배열에, 값에 해당하는 인덱스에 삽입한 후 뒤에서 부터 추출하여 새로운 배열에 삽입

<br>

### 2. Stable

<br>

### 3. 최선/평균/최악 - O(`n`)

<br>

### 4. 0이상 정수로의 표현 가능해야 사용이 가능

<br><br>

## 팀 정렬, Tim Sort

### 1. [합병 정렬](https://www.notion.so/jarvis08/Overall-of-Sorting-Algorithms-f0c43eb0f1134b7eaf9196ebccb6059f#2baf81ef347743adb3f47b41b6b6fbee)과 [삽입 정렬](https://www.notion.so/jarvis08/Overall-of-Sorting-Algorithms-f0c43eb0f1134b7eaf9196ebccb6059f#5ef527abfd804cb68125d3419d62c320)을 혼합한 방법

2002년에 파이썬으로 최초구현됐다. [합병 정렬](https://www.notion.so/jarvis08/Overall-of-Sorting-Algorithms-f0c43eb0f1134b7eaf9196ebccb6059f#2baf81ef347743adb3f47b41b6b6fbee)은 원소의 개수가 적을 경우 오버헤드가 발생하기 때문에 파티션 크기가 특정 값 이하(보통 16 또는 32)가 되면 [삽입 정렬](https://www.notion.so/jarvis08/Overall-of-Sorting-Algorithms-f0c43eb0f1134b7eaf9196ebccb6059f#5ef527abfd804cb68125d3419d62c320)을 사용한다.

하지만 일반적인 [합병 정렬](https://www.notion.so/jarvis08/Overall-of-Sorting-Algorithms-f0c43eb0f1134b7eaf9196ebccb6059f#2baf81ef347743adb3f47b41b6b6fbee)과 달리, 미리 어느정도 정렬 된 열(run)을 분할 하여 합병한다. 2개의 run에서 정렬된 부분(mergeLo, mergeHi)을 반복하여 합병하는 방식이다. run의 크기는 [삽입 정렬](https://www.notion.so/jarvis08/Overall-of-Sorting-Algorithms-f0c43eb0f1134b7eaf9196ebccb6059f#5ef527abfd804cb68125d3419d62c320)의 가장 빠른 범위(MIN_MERGE; 32개 이상의 요소)를 동적으로 계산하여 구성하며, 2개의 run으로 구성함(Binary Sort)으로 해서 삽입정렬의 횟수를 줄인다(삽입의 위치를 바이너리 서치로 탐색).

[합병 정렬](https://www.notion.so/jarvis08/Overall-of-Sorting-Algorithms-f0c43eb0f1134b7eaf9196ebccb6059f#2baf81ef347743adb3f47b41b6b6fbee)과 비슷한 특징을 가지고, 대부분의 경우 더 빠르며, 가장 많이 사용되는 정렬 중 하나이다.

<br><br>

### 2. Stable

<br>

### 3. 최선 - O(`n`)

최상의 경우 [삽입 정렬](https://www.notion.so/jarvis08/Overall-of-Sorting-Algorithms-f0c43eb0f1134b7eaf9196ebccb6059f#5ef527abfd804cb68125d3419d62c320)의 O(n)를 복잡도를 가지고, 최악의 경우 [합병 정렬](https://www.notion.so/jarvis08/Overall-of-Sorting-Algorithms-f0c43eb0f1134b7eaf9196ebccb6059f#2baf81ef347743adb3f47b41b6b6fbee)의 O(n logn)를 가진다.

<br>

### 4. 평균/최악 - O(`n logn`)

<br><br><br>

# O(n logn)

## 셸 정렬, Shell Sort

### 1. 입력 파일을 여러 개의 부분 파일로 세분화 후 부분 파일들을 [삽입 정렬](https://www.notion.so/jarvis08/Overall-of-Sorting-Algorithms-f0c43eb0f1134b7eaf9196ebccb6059f#5ef527abfd804cb68125d3419d62c320)

[삽입 정렬](https://www.notion.so/jarvis08/Overall-of-Sorting-Algorithms-f0c43eb0f1134b7eaf9196ebccb6059f#5ef527abfd804cb68125d3419d62c320)을 개선한 알고리즘이며, 방법에 따라 순차적으로 정렬하는 과정을 반복한다.

- Gap

  우선적으로 데이터를 띄엄띄엄 정렬시켜 둔다.

  1. gap 만큼 떨어진 데이터들을 부분정렬
  2. gap은 N/2 ~ 1
  3. gap = 1일 경우 데이터 전체를 [삽입 정렬](https://www.notion.so/jarvis08/Overall-of-Sorting-Algorithms-f0c43eb0f1134b7eaf9196ebccb6059f#5ef527abfd804cb68125d3419d62c320)한 것과 같다.

<br>

### 2. Stable

<br>

### 3. 최선 - O(`n logn`)

<br>

### 4. 평균 - O(`n (logn)^2`)

<br>

### 5. 최악 - O(`n (logn)^2`)

<br><br>

## 합병 정렬, Merge Sort

### 1. 배열을 쪼갠 후, 쪼개어진 이미 정렬되어 있는 2개의 파일을 하나의 정렬된 파일로 만들며 전체를 정렬

분할 과정: 배열을 반으로 쪼개며, 배열의 크기가 0이나 1이 될 때 까지 쪼갠다.

- 합병 과정

1. 2개의 리스트의 값들을 처음부터 하나씩 비교하여 두 개의 리스트의 값 중에서 더 작은 값을 새로운 리스트(sorted)로 옮긴다.
   1. 둘 중에서 하나가 끝날 때까지 이 과정을 되풀이한다.
   2. 만약 둘 중에서 하나의 리스트가 먼저 끝나게 되면 나머지 리스트의 값들을 전부 새로운 리스트(sorted)로 복사한다.
   3. 새로운 리스트(sorted)를 원래의 리스트(list)로 옮긴다.

<br>

### 2. Stable

<br>

### 3. 최선/평균/최악 - O(`n logn`)

분할 과정을 n번 거쳐야 크기가 1인 배열로 쪼갤 수 있으며, 분할 별로 합병을 진행

평균적인 [퀵 정렬](https://www.notion.so/jarvis08/Overall-of-Sorting-Algorithms-f0c43eb0f1134b7eaf9196ebccb6059f#b092ea76459e4ec1abdbb063a37dc159)보다 느리며, 데이터 크기 만큼의 메모리가 더 필요하므로 **공간복잡가 크다**.

<br>

### 4. 크기가 1인 배열로 분할하는 과정에 O(`logn`) 소요

<br><br>

## 퀵 정렬, Quick Sort

### 1. 분할 정복을 응용하지만, [합병 정렬](https://www.notion.so/jarvis08/Overall-of-Sorting-Algorithms-f0c43eb0f1134b7eaf9196ebccb6059f#2baf81ef347743adb3f47b41b6b6fbee)과는 다르게 Pivot을 이용하여 비균등 분할을 하며, 분할과 동시에 정렬을 진행

평균적으로 가장 좋은 성능을 내며, 스택 구조와 순환적 알고리즘을 사용한다.

- 과정 
  1. 피봇 값 설정
  2. 피봇 보다 작은 값은 피봇 왼쪽에, 큰 값은 피봇 오른쪽으로 이동
  3. 피봇 제외하고 왼쪽리스트, 오른쪽 리스트에 대해 재귀적으로 수행
  4. 부분 리스트를 더이상 분할할 수 없을 때까지 반복

가장 많이 구현되는 정렬 알고리즘 중 하나이며, `C`, `C++`, `PHP`, `Java`등 거의 모든 언어에서 제공하는 정렬 함수에서 퀵 정렬 혹은 퀵 정렬의 변형 알고리즘을 사용한다.

방식은 먼저 적절한 원소 하나를 기준(`피벗 ,pivot`)으로 삼아 그보다 작은것을 앞으로 빼내고 그 뒤에 피벗을 옮겨 피벗보다 작은 것, 큰 것으로 나눈다.

그리고 각각에서 다시 피벗을 잡고 정렬해서 각각의 크기가 0이나 1이 될 때까지 정렬한다.

- 피벗 잡는 방법.
  1. 난수(Random Quick Sort), 중위법 
     - 가장 쉽고 비효율적인 방법.
  2. 배열 중 3~9개의 원소를 골라서 이들의 중앙값을 피벗으로 고르는 것
     - Visual c++과 gcc에서 구현하고 있는 방법
     - 최악의 경우가 나올 수 있지만, 그 경우가 극히 드물게 된다.
  3. 인트로 정렬 
     - 재귀 깊이가 어느 제한 이상으로 깊어질 경우 [힙 정렬](https://www.notion.so/jarvis08/Overall-of-Sorting-Algorithms-f0c43eb0f1134b7eaf9196ebccb6059f#90ad70185a554e6c94bb67782286aad2) 알고리즘을 사용하여 항상 O(n log n)을 보장해주는 방법

<br>

### 2. Unstable

<br>

### 3. 최선/평균 - O(`n logn`)

<br>

### 4. 최악 - O(`n^2`)

<br>

### 5.  피벗이 계속해서 최소/최대값으로 지정될 때

<br><br>

## 힙 정렬, Heap Sort

### 1. 모든 원소를 힙(완전 이진 트리 구조)에 삽입한 후 root부터 삭제하며 새로운 배열에 저장

완전 이진 트리를 이용하고, 일정한 기억장소만을 필요로 한다.

- 최소힙/최대힙 트리를 구성하여 정렬하는 방법. 
  1. build heap
  2. root부터 원소 삭제하며 새로운 배열에 저장
- **자료 전체보다는 최대/최소값 부터 몇 개의 요소를 구할 때 유용**

기본적인 알고리즘은 `[선택 정렬](<https://www.notion.so/jarvis08/Overall-of-Sorting-Algorithms-f0c43eb0f1134b7eaf9196ebccb6059f#8d7c37dd7a614b45b93f9c244df68051>)`과 동일하다. 하지만 `힙 정렬`의 경우 최소값, 최댓값을 찾을때 배열을 순회하는게 아니라 만들어둔 힙을 사용한다. 따라서 요소를 조회하는 것에 `O(log n)`이 소요되며, 정렬하는 것에는 `O(n logn)`이 소요된다.

힙정렬은 추가적인 메모리를 전혀 필요로 하지 않는다. 또한 최악의 경우 `O(N^2)`의 성능을 내는 퀵정렬과는 다르게 `항상 일정한 성능`을 발휘하므로 최소한의 알고리즘만 정의할 경우 힙 정렬이 보다 안정적인 성능을 갖는다.

`[퀵 정렬](<https://www.notion.so/jarvis08/Overall-of-Sorting-Algorithms-f0c43eb0f1134b7eaf9196ebccb6059f#b092ea76459e4ec1abdbb063a37dc159>)`은 배열을 사용하므로, 대개 원소들끼리 근접한 메모리 영역을 사용한다. 하지만 `힙 정렬`의 경우 원소들이 보다 흩어져 있는 경우가 많아 `캐시 친화도`가 떨어진다.

또한 `힙정렬`은 일반적으로 `포인터 연산`을 많이 하기 때문에 `오버헤드가 크다`.

<br>

### 2. Unstable

<br>

### 3. 최선/평균/최악 - O(`n logn`)

<br>

### 4. 노드 하나를 삽입/삭제할 때 O(`logn`)을 소요

<br><br><br>

# O(n^2)

## 선택 정렬, Selection Sort

### 1. 첫 번째 순서에는 전체 배열의 최솟값을 삽입하며, 다음 인덱스에 남은 값 중에서의 최소값을 삽입

순서 별로 원소를 삽입할 인덱스가 정해져 있으며, 어떤 원소를 삽입할 것인지 선택하는 알고리즘이다. 모든 요소를 훑어서 가장 작은 요소를 골라내는 방식을 n번 반복한다.

배열의 상태와 무관하게 `n(n-1)/2` 에 비례하는 시간이 소요되며, **일반적으로 버블 정렬보다 약 두 배 빠르다**. 단 하나의 배열에서만 진행하므로 공간복잡도는 O(n)이다.

<br>

### 2. Unstable

<br>

### 3. 최선/평균/최악 - O(`n^2`)

<br>

### 4. 최악/최상 어떤 경우이든 요소 전부를 비교

<br><br>

# Reference

[진호박's Life Style](https://jinhobak.tistory.com/221)

[groonngroo](https://groonngroo.tistory.com/37)