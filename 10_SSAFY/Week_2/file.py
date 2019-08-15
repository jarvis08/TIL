# file.py
import os

# 디렉토리 내부 파일/디렉토리 조사
print(os.listdir())
print(len(os.listdir()))

# 파일명/디렉토리명 변경
# os.rename(현재 파일명, 바꿀 파일명)

"""
# pyformat 방법
for i in range(100):
    os.system('touch ./example/example_{}.txt'.format(i))

# f string : 삽입법
## python3.6부터 가능하며, SW test 불가
for i in range(100):
    os.system(f'touch ./example/example_{i}.txt')

# 더 기초적인 방법
os.chdir('example')
for i in range(100):
    os.system('touch example'+ str(i) + '.txt')
"""

# file명 한꺼번에 바꾸기
os.chdir('example')
files = os.listdir()

for name in files:
    # os.rename(name, 'Samsung_' + name)
    # renamed = name.replace('Samsung', 'SSAFY')
    # os.rename(name, renamed)
    os.rename(name, name.replace('Samsung', 'SSAFY'))