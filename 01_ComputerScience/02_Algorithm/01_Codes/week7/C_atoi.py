# 문자열 분해하여 unicode 변환 후 요소별로 자리수만큼 10을 곱하여 요소들을 더함
s_num = '-53'
l_uni = []
sign = 1
if '-' in s_num:
    sign = -1
    s_num = s_num[1:]
for n in s_num:
    l_uni += [ord(n) - 48]

summed = 0
for i in range(len(l_uni)):
    summed += (10**(len(l_uni) - i - 1) * l_uni[i]) * sign
print(summed)