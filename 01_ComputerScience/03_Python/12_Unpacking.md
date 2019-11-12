# Unpacking (*)

Asterisk(`*`)는 그 쓰임새가 다양한데, 그 중 하나는 Unpacking 연산자입니다. Unpacking 연산자란, 컨테이너 타입의 `[]`, `()`를 분해하여 요소들만을 꺼내올 수 있도록 합니다.

<br>

### 예제

```python
head, *rest = [20, 30, 40, 50]
print(head, rest)
# 20 [30, 40, 50]
*rest, tail = [20, 30, 40, 50]
print(rest, tail)
# [20, 30, 40] 50
head, *middle, *tail = [20, 30, 40]
print(head, middle, tail)
# 20 [30] 40
```

[참고자료](https://dimitrisjim.github.io/python-unpackings-unpacked.html)

<br>

<br>

## 응용

### 같은 요소를 갖는 리스트 제거하기

```python
print(overlapped)
eliminated = []
for alist in overlapped:
  print('\nalist: ', alist)
  print('*alist: ', *alist)
  if {*alist} not in eliminated:
    print('{*alist}: ', {*alist})
    eliminated.append({*alist})
result = list(map(list, eliminated))
print(result)
```

```bash
[['fradi', 'abc123'], ['abc123', 'fradi'], ['frodo', 'abc123']]

alist:  ['fradi', 'abc123']
*alist:  fradi abc123
{*alist}:  {'fradi', 'abc123'}

alist:  ['abc123', 'fradi']
*alist:  abc123 fradi
{*alist}:  {'abc123', 'fradi'}

alist:  ['frodo', 'abc123']
*alist:  frodo abc123
{*alist}:  {'frodo', 'abc123'}
[['fradi', 'abc123'], ['frodo', 'abc123']]
```

각 리스트들을 set으로 변환 후, 같은게 있는지 비교합니다. set은 요소의 순서가 달라도, 구성이 같다면 같은 것으로 취급합니다.