# SSAFY_Week3_Day5

 **참고자료** : ./50_SSAFY/8ython/notes/

---

- 문자열 덧셈 하기

  #### 문자열 조작 및 반복/조건문 활용[¶](http://localhost:8888/notebooks/problems/problem04.ipynb#문자열-조작-및-반복/조건문-활용)

  **문제 풀기 전에 어떻게 풀어야할지 생각부터 해봅시다!**

> 사람은 덧셈을 할때 뒤에서부터 계산하고, 받아올림을 합니다.
>
> 문자열 2개를 받아 덧셈을 하여 숫자를 반환하는 함수 `my_sum(num1, num2)`을 만들어보세요.

```python
def my_sum(num1, num2):
    n1 = list(map(int, num1))
    n2 = list(map(int, num2))
    # 자리수 맞추기
    while True:
        if len(n1) > len(n2):
            n2.insert(0, 0)
        elif len(n1) < len(n2):
            n1.insert(0, 0)
        else:
            break

    # 뒤부터 더해서 올림받기
    up = False
    result = []
    for i in range(1, len(n1)+1):
        tmp = n1[-i] + n2[-i]
        if up:
            tmp += 1
            up = False        
        if tmp >= 10:
            if i == len(n1):
                result.append(f'{tmp}')
                continue
            up = True
            result.append(f'{tmp-10}')
        else:
            result.append(f'{tmp}')
    result.reverse()
    return int(''.join(result))
```



