# SSAFY_Week3_Day5

 **참고자료** : ./50_SSAFY/8ython/notes/

---

## Iterable에 사용가능한 함수

### `map()`, `filter()`, `zip()` 이외의 함수들

- `all()` : all은 인자로 받는 `iterable`(`range`, `list` 등)의 **모든 요소가 참**이거나 **비어있으면** `True`를 반환

  ```python
  print(all([1, 2, 5, '6']))
  print(all([[], 2, 5, '6']))
  """result
  True
  False"""
  
  # all() 만들어보기
  def my_all(x):
      for element in x:
          # 하나라도 False면 의미X
          # [[], 1] 의 경우 빈 list일지라도 존재하기 때문에 True로 처리
          if not element:
              return False
      return True
  ```

- `any()` : any는 인자로 받는 `iterable`(`range` 혹은  `list`)의 요소 중 하나라도 참이면 True를 반환하고, 비어있으면 False를 반환

  ```python
  print(any([1, 2, 5, '6']))
  print(any([[], 2, 5, '6']))
  print(any([0]))
  """result
  True
  True
  False"""
  
  # any() 만들어보기
  def my_any(x):
      for element in x:
          if element:
              return True
      return False
  ```

- 소수 찾기

  ```python
  numbers = [26, 39, 51, 53, 57, 79, 85]
  
  def sosu(l):
      for num in l:
          for i in range(2, num):
              if num % i == 0:
                  print(f'{num}은(는) 소수가 아닙니다. {i}는 {num}의 인수입니다.')
                  break
          else:
              print(f'{num}은(는) 소수입니다.')
              
  sosu(numbers)
  """python
  26은(는) 소수가 아닙니다. 2는 26의 인수입니다.
  39은(는) 소수가 아닙니다. 3는 39의 인수입니다.
  51은(는) 소수가 아닙니다. 3는 51의 인수입니다.
  53은(는) 소수입니다.
  57은(는) 소수가 아닙니다. 3는 57의 인수입니다.
  79은(는) 소수입니다.
  85은(는) 소수가 아닙니다. 5는 85의 인수입니다."""
  ```

- 최대공약수, 최소공배수 구하기

  ```python
  # 유클리드 호제법
  # GCD = Greatest Common Divisor
  # LCM = Least Common Multiple
  
  # GCD/LCM 구하기
  def gcdlcm(a, b):
      # max, min을 할 필요 없음
      # 어차피 작은 수를 큰 수로 나누면 나머지는 작은수
      # m, n = max(a, b), min(a, b)
      m, n = a, b
      while n > 0:
          m, n = n, m % n
      return [m, int(a*b / m)]
  print(gcdlcm(3, 12))
  print(gcdlcm(1071, 1029))
  
  
  # 재귀함수로 GCD 구하기
  def gcd(n, m):
      if n % m == 0:
          return m
      else:
          return gcd(m, n%m)
  
  def gcdlcm2(n, m):
      g = gcd(n, m)
      l = n*m // g
      return g, l
  
  print(gcdlcm2(3, 12))
  print(gcdlcm2(1071, 1029))
  
  """result
  [3, 12]
  [21, 52479]
  (3, 12)
  (21, 52479)"""
  ```

- 과일 개수 골라내기

  ```python
  basket_items = {'apples': 4, 'oranges': 19, 'kites': 3, 'sandwiches': 8}
  fruits = ['apples', 'oranges', 'pears', 'peaches', 'grapes', 'bananas']
  
  def fruits_checker(d):
      cnt = 0
      non_cnt = 0
      for k, v in d.items():
          if k in fruits:
              cnt += v
          else:
              non_cnt += v
      return cnt, non_cnt
  fruits_checker(basket_items)
  ```

- 절대값 함수 만들기

  절대값은 숫자(int, float)가 들어오면 절대값을 반환하고, 복소수(complex)가 들어오면 그 크기를 반환

  - 복소수 크기

    : 복소 평면이라는 x축이 실수부, y축이 허수부인 그래프에서 3+4j라면 (3,4)인 지점의 벡터거리(원점으로부터 직선 거리)가 복소수의 크기

  ```python
  # 방대승님
  def my_abs(x):
      return (x.real**2+x.imag**2)**0.5
  
  # 오재석님
  def my_abs(x):
      # conjugate는 켤레복소수를 제곱 제곱 -1 하여 반환
      return (x * x.conjugate()).real**0.5
  
  갓동주님 명세서 따르기 코드
  def my_abs(x):
      if type(x) == type(1j):
          return (x.real**2+x.imag**2)**0.5
      else:
          if x == 0:
              return x ** 2
          elif x < 0:
              return x * -1
          else:
              return x
  
  print(abs(3+4j), abs(-0.0), abs(-5))
  """result
  5.0 0.0 5"""
  ```

- 문자열 덧셈 하기

  #### 문자열 조작 및 반복/조건문 활용[¶](http://localhost:8888/notebooks/problems/problem04.ipynb#문자열-조작-및-반복/조건문-활용)

  **문제 풀기 전에 어떻게 풀어야할지 생각부터 해봅시다!**

  > 사람은 덧셈을 할때 뒤에서부터 계산하고, 받아올림을 합니다.
  >
  > 문자열 2개를 받아 덧셈을 하여 숫자를 반환하는 함수 `my_sum(num1, num2)`을 만들어보세요.

  **절대로 return int(num1)+int(num2) 이렇게 풀지 맙시다!!**

  **재귀함수도 사용할 필요 없습니다.**

---

- hw_ws dirctory를 Gitlab repository로 옮기기

  ```shell
  # 이미 origin이 있기 때문에 gitlab을 다시 origin으로 설정해주는 작업을 시작
  $ git remote add origin https://lab.ssafy.com/mtice/cho_dong_bin.git
  fatal: remote origin already exists.
  
  
  $ git remote -v
  origin  https://github.com/jarvis08/hw_ws.git (fetch)
  origin  https://github.com/jarvis08/hw_ws.git (push)
  
  git remote remove origin
  $ git remote add origin https://lab.ssafy.com/mtice/cho_dong_bin.git
  
  $ git remote -v
  origin  https://lab.ssafy.com/mtice/cho_dong_bin.git (fetch)
  origin  https://lab.ssafy.com/mtice/cho_dong_bin.git (push)
  
  $ git push -u origin master
  
  $ git remote add github https://github.com/jarvis08/hw_ws.git
  $ git remote -v
  github  https://github.com/jarvis08/hw_ws.git (fetch)
  github  https://github.com/jarvis08/hw_ws.git (push)
  origin  https://lab.ssafy.com/mtice/cho_dong_bin.git (fetch)
  origin  https://lab.ssafy.com/mtice/cho_dong_bin.git (push)
  
  # origin을 gitlab으로 교체했기 때문에 origin master는 gitlab
  $ git push origin master
  # github이라는 이름으로 추가했기 때문에 github master로 push
  $ git push github master
  
  ```

  ```shell
  # 로그인 정보 잘 못 입력했을 때
  git credential reject
  protocol=https
  host=github.com
  ```

- CSV = Comma Seperated Values

  - `csv.writer(파일명)` : '파일명'을 조작하는 writer를 원하는 이름으로 생성

    `writer.writerow(iterable_item)` : iterable_item을 전에 생성한 writer를 사용하여 한줄씩 끊어서 작성(writerow)

  - `csv.DictWriter(파일명, fieldnames=필드명_list)`

    이 함수를 선언시, `writer.writerow(iterable)`을 통해 dictionary를 그대로 넣어도 csv 형태 작성

    `writer.writeheader()` : 해당 코드를 넣어두면 fieldnames까지 첫 줄에 작성

  ```python
  # csv 파일 만들기
  lunch = {
      '진가와' : '01011112222',
      '대우식당' : '01054813518',
      '바스버거' : '01088465846'
  }
  # 1. lunch.csv 데이터 저장
  with open('lunch.csv', 'w', encoding='UTF-8') as f:
      for k, v in lunch.items():
          f.write(f'{k},{v}\n')
  
  # 2. ',' join을 사용하여 string 만들기
  with open('lunch.csv', 'w', encoding='UTF-8') as f:
      for item in lunch.items():
          f.write(','.join(item))
  
  # 3. csv 라이브러리 사용
  # writer와 reader가 따로 존재
  import csv
  with open('lunch.csv', 'w', encoding='utf-8', newline='') as f:
      csv_writer = csv.writer(f)
      for item in lunch.items():
          csv_writer.writerow(item)
          
  # 4. csv.DictWriter()
  # field name을 설정 가능
  with open('student.csv', 'w', encoding='utf-8', newline='') as f:
      fieldnames = ['name', 'major']
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerow({'name':'john', 'major':'cs'})
      writer.writerow({'name':'dongbin', 'major':'ie'})
  ```

  