# Algorithm

---

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

  

