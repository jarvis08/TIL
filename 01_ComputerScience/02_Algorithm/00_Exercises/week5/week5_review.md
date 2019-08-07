# Week5 Review

---

- min_max.py

  차례대로 max, min 값과 비교하며 값을 갱신

- electric_bus.py

  1. 이전 위치(`pre`)를 현재 위치(`cur`) 값으로 할당
  2. 전체 `while`문을 통해 현재 위치가 종료 지점 이상의 거리를 이동했는가를 탐색

  2. 현재 위치 변수를 최대 이동거리(`K`) 만큼 이동
  3. `for`문을 이용하여 이동한 지점으로부터 이전 지점과 현재 지점 사이에 충전소가 있는가를 탐색(이동한 지점으로부터 `-1`씩 이동)
  4. 만약(`if`) 충전소를 찾은 지점이 `pre`의 위치와 동일하다면 충전횟수를 `0`으로 하며 `break`

- cards.py

  1. Counting Sort 방식처럼 카드의 숫자들을 인덱스로 하여 count

  2. max index를 `0`으로 설정 후 비교(`<=` 혹은 `>=`이용)해 가며 max index를 구한다.
  3. max index의 index가 개수가 최대인 카드의 숫자

  - 띄워지지 않은 한 자리 숫자의 나열을 input으로 받을 때

    ```python
    cards = input()
    # 숫자 별 개수 구하기
    cnt = [0] * 10
    for i in range(len(cards)):
        cnt[int(cards[i])] += 1
    ```

    ```python
    cards = input()
    card_list = []
    # 숫자 별 개수 구하기
    for i in range(len(cards)):
        card_list.append(int(cards[i]))
    ```

- partial_sum.py

  하나의 요소 별로 part 크기만큼의 횟수로 사용됨

  ```python
  list = [1, 2, 3, 4, 5, 6, 7]
  ```

  - 3개씩의 합들 중 max 값을 구할 때, index 2의 숫자 3은 3회 사용된다.

    (1, 2, `3`) (2, `3`, 4) (`3`, 4, 5)

    이를 생각해 봤을 때, 세번씩 계속 더하는건 효율 적이지 않다.

    Do not recompute!

  - 해결 방법 2가지

    - 방법 1

      1. (1, 2, 3) = 6
      2. (2, 3, 4) = 6 - 1 + 4 = 9
      3. (3, 4, 5) = 8 - 2 + 5 = 11

    - 방법 2, Sliding Window

      (4, 5, 6)까지의 합을 구하려면 [(1~6) - (1~3)]

- flatten.py

  - 큰값 - 1, 작은 값 + 1로 인해 data의 status가 계속 변경된다.

  - dump 횟수가 충분히 크며 전체 개수가 column 수에 비례하다면, `전체 개수 / column 개수`

  - max, min을 구했을 때 그 둘의 차이가 1 혹은 0 이면 종료

  - 상위 level 방법

    Counting Sort를 응용하여 count 기반의 dump