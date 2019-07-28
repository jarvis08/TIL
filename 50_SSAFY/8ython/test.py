# global 선언 후 수정
global_num = 10
def funct():
    global global_num
    global_num = 5
funct()
print(global_num)