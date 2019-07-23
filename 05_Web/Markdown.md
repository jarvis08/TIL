# :+1: Markdown Language

`\` : 뒤의 단어를 명령어가 아닌 일반 기호로 처리(backslash, excape)(python \n처럼)

`(-, *, +) + Enter` : black point, white point

``` `내용` ``` : `코드블록에 내용 넣기(backtick)`

```python
# backtick을 결과에 포함하고 싶을 때
A backtick-delimited string in a code span: `` `foo` ``
결과 :: A backtick-delimited string in a code span: `foo`
```

---

`*italic*` : *italic*

`_italic_` : _italic_

`**bold**` : **bold**

`__bold__` : __bold__

`~~strike through~~` : ~~strike through~~

---

```PYTHON
​```python + Enter : 블록 코드(3 back tick)
(Ctrl + Enter) : 빠져나가기
```

---

`--- + Enter` : 분리 선 긋기

`> ` : 단 띄우기

---

```python
# image 처리하기
![Alt text](/path/to/img.jpg)
![Alt text](/path/to/img.jpg "Optional title")
[img id]: url/to/image  "Optional title attribute"
# Reference-style image syntax
![Alt text][img id]
```

```python
# source image를 center에 정렬하여 표시
<center><img src="./images/01/variable.png", alt="variable"/></center>
```

---

`[Google]: http://google.com/` : [Google]: http://google.com/ 링크 생성

`<http://example.com/>` : <https://github.com/jarvis08>

`<address@example.com>` : <cdb921226@gmail.com>

`https://github.com/jarvis08` : https://github.com/jarvis08 , GFM(Github Flavored Markdown)은 그냥 생성

`This is [an example](https://github.com/jarvis08 "My Github") inline link.` : This is [an example](https://github.com/jarvis08 "My Github") inline link.

---

```python
# 표 그리기
| Left | Center | Right |
|:-----|:------:|------:|
|aaa   |bbb     |ccc    |
|ddd   |eee     |fff    |

 A | B 
---|---
123|456

A |B 
--|--
12|45

# 표 그리기 예시
|<center>예약문자</center>|내용(의미)|
|:--------:|:--------:|
|\n|줄바꿈|
|\t|탭|
|\r|캐리지리턴|
|\0|널(Null)|
|`\\`|`\`|
|\'|단일인용부호(')|
|\"|이중인용부호(")|
```

---

```python
# Block level HTML 그대로 사용 가능
<table>
    <tr>
        <td>Foo</td>
    </tr>
</table>
```

---

https://github.com/jarvis08/emoji-cheat-sheet

https://www.webfx.com/tools/emoji-cheat-sheet/

`:+1:` : :+1:

`:thumbsup:` : :thumbsup:

`:-1:` : :-1:

`:shit:` : :shit:

`:question:` : :question:

`:exclamation:` : :exclamation:

`:grey_exclamation:` : :grey_exclamation:

`:boom:` : :boom:

`:star:` : :star:

`:metal:` : :metal:

`:clap:` : :clap:

`:fu:` : :fu:

### Language :	Python	/	C++	/	C

Project Experience :	True	/	False	/	False

### OS :	Windows	/	Linux	/	OS X

Project Experience :	True	/	True	/	False

### DL Framework :	Tensorflow	/	Pytorch

Project Experience :	True	/	False