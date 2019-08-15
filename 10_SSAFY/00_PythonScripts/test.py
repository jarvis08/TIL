#-*-coding: utf-8
class Point:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Square:
    def __init__(self, point_1, point_2):
        self.p1 = point_1.x , point_1.y
        self.p2 = point_2.x, point_2.y
    
    def get_area(self):
        return abs(self.p1[0] - self.p2[0]) * abs(self.p1[1] - self.p2[1])
    
    def __repr__(self):
        isit = False
        if abs(self.p1[0] - self.p2[0]) == abs(self.p1[1] - self.p2[1]):
            isit = True
        return 'p1과 p2를 이용한 사각형은 정사각형인가? {}'.format(isit)

p1 = Point(3, 4)
p2 = Point(6, 8)
s1 = Square(p1, p2)
print(s1.get_area())
print(s1)