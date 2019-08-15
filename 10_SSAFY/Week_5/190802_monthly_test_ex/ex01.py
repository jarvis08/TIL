# 파일명 및 함수명을 변경하지 마시오.
def positive_sum(numbers):
    summed = 0
    for number in numbers:
        if number > 0:
            summed += number
    return summed





# 실행 결과를 확인하기 위한 코드입니다. 수정하지 마시오.
if __name__ == '__main__':
    print(positive_sum([1, -4, 7, 12])) #=> 20
    print(positive_sum([-1, -2, -3, -4])) #=> 0