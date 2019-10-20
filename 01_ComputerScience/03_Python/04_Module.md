# Module

 **참고자료** : ./50_SSAFY/8ython/notes/05.module.ipynb

## Module, 모듈

- 정의 : 기능별 단위로 분할한 부분

- 모듈화 목적 : 복잡성을 감소시키기 위해 단순화, abstraction(요약)을 하는 수단 중 하나

- 함수 : 가장 기본적인 모듈

- `import` : 모듈을 활용하기 위해서는 반드시 `import` 문을 통해 내장 모듈을 namespace로 불러와야 한다.

  ```python
  # hello.py
  def hi():
      return '안녕하세요'
  
  # module.ipynb
  import hello
  print(hello.hi())
  dir(hello)
  """result
  안녕하세요
  ['__builtins__',
   '__cached__',
   '__doc__',
   '__file__',
   '__loader__',
   '__name__',
   '__package__',
   '__spec__',
   'hi']"""
  ```

<br>

### Package 만들기

- module의 상/하 구조를 나누어 구조화 할 수 있도록 계층화

- `__init__.py`

  : 파이썬이 directory를 package로 취급하게 만들기 위해 필요

  : string 처럼 흔히 쓰는 이름의 directory가 의도하지 않게 모듈 검색 경로의 뒤에 등장하는, 올바른 모듈들을 가리는 일을 방지하기 위함

  This prevents directories with a common name, such as string , unintentionally hiding valid modules that occur later on the module search path

```python
/myPackage
    __init__.py
    /math
        __init__.py
        formula.py
    /web
        __init__.py
        url.py

# __init__.py
pi = 3.145

# formula.py
def pi():
    return 3.14
```

---

- `from` 모듈 `import` `attribute`

  `attribute` : 변수(데이터), 함수, 클래스 등을 모두 포함하는 '속성' 값

  ```python
  # formula.py
  from myPackage.math import formula
  
  # math directory의 __init__.py
  from myPackage import math
  
  # __init__.py의 변수 pi
  from myPackage.math import pi
  
  print(math.formula.pi())
  print(math.pi)
  print(pi)
  """result
  3.14
  3.145
  3.145"""
  ```

  ```python
  # 외장 함수 math
  import math
  # 직접 만든 myPackage
  # math까지만 불러오면 외장 함수 math를 덮어씀
  from myPackage.math.formula import pi
  
  print(math.sqrt(4))
  print(pi())
  """result
  2.0
  3.14"""
  ```

  ---

- `myPackage/__init__.py` 안에

  ```python
  from .math import formula
  ```

  기입 시

  ```python
  import myPackage
  ```

  만으로도 `formula.pi()`를 사용 가능

  ```python
  # __init__.py 수정 전
  import myPackage
  dir(myPackage)
  """result
  ['__builtins__',
   '__cached__',
   '__doc__',
   '__file__',
   '__loader__',
   '__name__',
   '__package__',
   '__path__',
   '__spec__']"""
  
  # __init__.py 내부에 from .math import formula 작성 후
  """result
  ['__builtins__',
   '__cached__',
   '__doc__',
   '__file__',
   '__loader__',
   '__name__',
   '__package__',
   '__path__',
   '__spec__',
   'formula',
   'math']"""
  ```

  ---

- `.`을 이용하여 directory를 지정하지만, directory 지정이 `.` 사용의 목표가 아니다.

- `.`은 module의 계층을 구분하는 의미로써, 물리적 directory 구조를 생각하면 오류 발생

  - 따라서 `__init__.py` 내부에 하부 계층의 module을 `import` 해주지 않는 이상,

    `import myPackage` 만으로 하부 계층의 module(ex.`formula.py`)을 사용 불가

- `__all__=['모듈명', '모듈명']`

  `__init__.py`에 package의 사용할 모듈 모두 지정해 두기

  ```python
  # __init__.py에 추가하기
  __all__ = ['math','web']
  
  # 다른 곳에서 *를 사용하여 불러오기
  from myPackage import *
  """result
  ['__all__',
   '__builtins__',
   '__cached__',
   '__doc__',
   '__file__',
   '__loader__',
   '__name__',
   '__package__',
   '__path__',
   '__spec__',
   'math',
   'web']"""
  ```

<br>

### 기타

- python version 3.3 이후로는 `__init__.py`를 작성하지 않더라도 실행 가능

- 현재는 하위 호환성을 위해 계속 작성

- `from` 모듈 `import` attribute `as` 별칭

  : 내가 지정하는 이름으로 `import`

  ```python
  from bs4 import BeautifulSoup as bs
  ```


