class Person:
    name = ''
    def __init__(self, name):
        self.name = name

    
    def __repr__(self):
        return '대표하는거에용'

p1 = Person('동빈')
print(p1)
