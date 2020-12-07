# Vim Shortcuts

출처: [Outsider](https://blog.outsider.ne.kr/540)

Vi의 심화 버전인 Vim(Vi IMproved)의 단축키입니다.

- `:set encoding=utf-8` : `vim <파일명>`으로 실행했는데 한글이 깨져 보일 때

<br>

<br>

## 0. Default Mode

- `h`, `j`, `k`, `l`: 좌, 하, 상, 우 커서 이동
- `/단어`: 단어 검색
- `dd`: 현재 줄 잘라내기
- `dw`: 단어 잘라내기
- `cw`: 현재 커서부터 단어 잘라내기(단어의 앞쪽은 적용X)
- `yy`: 현재 줄 복사하기
- `p`: 붙혀넣기
- `r`: 현재 글자 교체하기
- `x`: 현재 글자 지우기
- `X`: 앞의 글자 지우기
- `s`: 현재 커서 글자 삭제 후 insert mode 실행
- `>`: 들여쓰기
- `<`: 내어쓰기
- `u`: Undo
- `Ctrl + r`: Redo
- `.`: 이전 명령어를 다시 실행
- `%`: 괄호 쌍 이동
- `#`: 현재 커서의 문자열을 윗 방향으로 탐색(`/문자열` 혹은 `?문자열`과 같은 효과)
- `*`: 현재 커서의 문자열을 아래 방향으로 탐색

<br>

### 0-0. Replace

vim에서 replace 할 때 쓰는 `:%s/old/new/g` 형태에서 `/` 는 고정된 delimiter가 아니라 %s 이후 첫 문자를 command 내 delimiter로 지정하겠다는 의미입니다.

즉,  `:%s@old@new@g` 해도 동일한 기능이며, `/` 가 포함된 string을 바꿀때 사용하면 용이합니다. 또한, sed 명령어도 동일하게 적용 됩니다.

- `1,3s/old/new/g`: 1번부터 3번 line 까지 교체
- `.,$s/old/new/g`: 현재 line 부터 끝까지 교체
- Split 된 화면들 중 하나의 화면에서 replacing 사용 후, 옆 split 화면으로 가서 `:%s` 라고만 치면 동일한 작업 진행

<br>

### 0-1. Go to Definition

- `\g`: go to definition
  - Kakao Internship 당시 사용했으나, 현재는 `ctags`를 사용하므로 `ctrl+]`
- `:e#`: go to definition 후, 다시 원래 보던 곳으로 돌아가기
  - `ctags`의 경우 `ctrl+t` 또한 동일한 동작

<br>

### 0-2. Split Window

- `:vs`: 가로 분할
- `:sv`: 세로 분할
- `Ctrl + w, w`: 커서(내 위치)를 분할 창들 중 바로 오른쪽 칸으로 이동
- `Ctrl + w, h/l`: 커서를 분할된 창들 중 왼/오른쪽으로 이동
- `Ctrl + w, r`: 현재 분할된 창을 오른쪽 칸으로 옮김

<br>

### 0-3. Fold

- `zc`: 코드 접기(fold)
- `zo`: 접힌 코드 펼치기
- `zd`: fold 지우기
- `zR`: 접힌 코드 모두 펼치기
- `zM`: 코드 모두 접기
- `zD`: 모든 fold 지우기

<br>

<br>

## 1. Move & Find

- `w`: 단어 첫글자로 이동하기
- `W`: 화이트스페이스 단위로 다음 글자로 이동하기
- `e`: 단어의 마지막 글자로 이동하기
- `E`: 화이트스페이스 단위로 단어의 마지막 글자로 이동하기
- `b`: 백워드 방향으로 단어의 첫글자로 이동하기
- `B`: 백워드 방향으로 화이트스페이스 단위로 다음 글자로 이동하기
- `ge`: 백워드 방향으로 단어의 마지막 글자로 이동하기
- `gg`: 문서 맨 앞으로 이동
- `G`: 문서 맨끝으로 이동
- `0`: 라인 맨 앞으로 이동
- `$`: 문장 맨 뒤로 이동
- `^`: 문장 맨 앞으로 이동(맨 앞의 Tab 뒤로)
- `x`: 커서가 위치한 char 하나를 제거
- `X`: 커서 왼쪽 char 하나를 제거
- `cc`: line 삭제 후 linsert mode 시작
- `cw`: word 삭제 후 insert mode 시작
- `s`: 커서가 위치한 char 하나를 제거한 후 insert mode 시작
- `S`: 커서가 위치한 line을 제거한 후 insert mode 시작
- `q + <a-z>`: record 시작 및 `esc`로 종료하며, 해당 알파벳에 해당 레코드를 저장
- `r + <한글자>`: 커서 아래의 char를 입력하는 하나의 char로 교체

<br>

### 1-2. Page

- `Ctrl + f`: 다음 페이지 이동
- `Ctrl + b`: 이전 페이지 이동
- `Ctrl + u`: 페이지절반만큼 다음으로 이동
- `Ctrl + d`: 페이지절반만큼 이전으로 이동
- `H`: 현재 화면의 맨 위라인으로 이동
- `M`: 현재 화면의 중간 라인으로 이동
- `L`: 현재 화면의 마지막 라인으로 이동

<br>

<br>

## 2. Find

### 2-1. Find

`/단어`: 문서에서 단어 찾기 n이나 N으로 다음/이전 찾기

`*`: 현재 단어를 포워드 방향으로 찾기

`#`: 현재 단어를 백워드 방향으로 찾기

<br>

### 2-2. Brace

- `%`: {}나 ()에서 현재 괄호의 짝으로 이동
- `]]`: 포워드 방향으로 여는 컬리 블레이스( { )로 이동
- `[[`: 백워드 방향으로 여는 컬리 블레이스( { )로 이동
- `][`: 포워드 방향으로 닫는 컬리 블레이스( { )로 이동
- `[]`: 백워드 방향으로 닫는 컬리 블레이스( { )로 이동

<br>

<br>

## 3. Insert Mode

- `i`: 현재 커서 위치에서 Insert Mode 실행
- `I`: 현재 커서의 라인 맨 앞에서 Insert Mode 실행
- `a`: 현재 커서 다음칸에 Insert Mode 실행
- `A`: 현재 커서의 라인 맨 뒤에서 Insert Mode 실행
- `O`: 윗줄에 빈 라인 생성 후 Insert Mode 실행
- `o`: 아랫줄에 빈 라인 생성 후 Insert Mode 실행

<br>

<br>

## 4. Visual Mode

- `v`: Visual Mode 실행 및 케릭터 단위 선택
- `V`: 줄(행) 단위 선택
- `ctrl+v`: 블록으로 선택(여러 줄, 공통 범위 선택에 용이)
- `y`: (Yank) 복사하기
- `c`: 잘라내기
- `p`: (Put)붙여넣기
- `J`: 다음 라인을 현재 줄의 끝으로 이어 붙힘
- `~` : 선택 문자 대소문자 변경
- `Ctrl + A` : 숫자를 증가시키기
- `Ctrl + X` : 숫자를 감소시키기

<br>

### 4-1. Combos

- `v, i, w`: 현재 단어 선택
- `(Visual Mode), 0`: 라인 맨 앞으로 드래그
- `(Visual Mode), shift+0`: 라인 맨 뒤로 드래그
- `(ctr+shift+v), I, 입력, <esc>`:  블록단위 선택하여 여러 줄에 한번에 입력하기

<br>

### 4-2. Command-line editing

- `:he cmdline-editing` : see details about command-line editing
- `Ctrl + r` : command-line editing mode start
- `(Ctrl + r) + (Ctral + w)` : put word below the current curser to the command line

<br>

<br>

## 5. Colon

- `:w`: 문서 저장하기
- `:q`: 현재 문서 닫기
- `:q!`: 저장하지 않고 닫기
- `:wq`: 저장하고 닫기
  - `ZZ`: 동일 기능
- `:숫자`: 지정한 라인넘버로 이동
- `:set number`: Editor 왼쪽에 line number 표시
- `:set nonumber`: line number 표기 제거
- `:sh`: shell로 돌아감 -> shell에서 `Ctral+d` 입력 시 vim으로 돌아옴

<br>

### 5-1. Tab & Division

`:new`: 가로로 분할된 새 창을 생성하여 열기

`:vs`: 세로로 분할된 새 창을 생성하여 열기

`Ctrl + w`: 분할창 간에 이동하기

`:tabnew`: 새로운 탭 생성하여 열기

`:gt`: 다음 탭으로 이동하기

`:gT`: 이전 탭으로 이동하기

`:e ./`: 현재 탭에 오픈할 파일 탐색하기( ./ 는 현재위치에서 탐색 시작)

`:colorscheme 스키마명`: VIM의 칼라스키마를 변경함(blue, desert, evening 등.. 스키마명에서 탭누르면 자동완성됨)

<br>

### 5-2. Buffer

`:buffers`: 현재 Vim에서 여러 파일을 열었을때 버퍼에 있는 목록 확인

`:buffer 숫자`: 버퍼 목록에 나온 숫자를 입력하면 해당 파일을 오픈함 (`:b` 도 가능)

`:bnext`: 버퍼에 있는 다음 파일로 이동 ( `:bn` 도 가능)

`:bprevious`: 버퍼에 있는 이전 파일로 이동 ( `:bp` 도 가능)

`:ball`: 버퍼 목록에 있는 파일들이 가로로 분할된 창에 열림

