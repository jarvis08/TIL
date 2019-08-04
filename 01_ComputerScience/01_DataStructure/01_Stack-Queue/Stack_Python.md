# Data Structure

---

- OOP_Exercise_Stack구현

  [Stack](https://ko.wikipedia.org/wiki/스택) : LIFO(Last in First Out)으로 구조화된 자료구조
  
  1. `empty()`: 스택이 비었다면 True을 주고, 그렇지 않다면 False가 된다.
  2. `top()`: 스택의 가장 마지막 데이터를 넘겨준다. 스택이 비었다면 None을 리턴한다.
  3. `pop()`: 스택의 가장 마지막 데이터의 값을 넘겨주고, 해당 데이터를 삭제한다. 스택이 비었다면 None을 리턴한다.
  4. `push()`: 스택의 가장 마지막 데이터 뒤에 값을 추가한다. 리턴 값은 없다.
  
  ```python
  class Stack:
      def __init__(self):
          self.data = []
      
      def push(self, n):
          self.data.append(n)
      
      def empty(self):
          # if self.data:
          #    return False
          # return True
          return not bool(self.data)
          
      
      def top(self):
          if self.data:
              return self.data[-1]
          # 함후가 아무런 return을 하지 않으면 알아서 None 처리하므로,
          # return None 필요 없다
      
      def pop(self):
          if self.data:
              last = self.data[-1]
              del self.data[-1]
              return last
      
      def __repr__(self):
          return f'{self.data}'
  
  stack = Stack()
  print(stack.empty())
stack.push(1)
  print(stack.empty())
  stack.push(2)
  stack.push(3)
  print(stack)
  stack.pop()
  print(stack.top())
  
  """result
  True
  False
  [1, 2, 3]
  2"""
  ```
  
  

