# Vim Plug-in

참고 자료: [junegunn](https://github.com/junegunn/vim-plug)

## vim-plug Installation

junegunn님이 개발하신 vim-plug를 통해 vim의 플러그인들을 관리합니다. 아래의 명령어로 설치합니다. 아래 명령어는 OS X용 명령어이며, Windows에서는 위 참고 자료에 적힌 github repository로 이동하여 참고하시면 됩니다.

```bash
$ curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
```

위 명령어를 통해 설치한 후, 아래와 같이 `.vimrc`를 생성하고, 원하는 plugin을 작성합니다.

```bash
$ vim ~/.vimrc
call plug#begin('~/.vim/plugged')
Plug 'tpope/vim-sensible'
Plug 'junegunn/seoul256.vim'
call plug#end()
```

작성한 후 `.vimrc`에서 빠져나오지 않은 채로 다음을 입력합니다.

```
:source %
:PlugInstall
```

입력 시 PlugInstall이 실행되며 설치가 완료되어 `Done`이라고 뜨는 것을 볼 수 있습니다.

<br>

### Updating plugins

Run `:PlugUpdate` to update the plugins. After the update is finished, you can review the changes by pressing `D` in the window. Or you can do it later by running `:PlugDiff`.

<br>

### Removing plugins

1. Delete or comment out `Plug` commands for the plugins you want to remove.
2. Reload vimrc (`:source ~/.vimrc`) or restart Vim
3. Run `:PlugClean`. It will detect and remove undeclared plugins.

<br>

<br>

## My Plug-in

[Vim Awesome](https://vimawesome.com/)에서 Vim Plugin들을 찾아볼 수 있습니다. Vim Vundle, vim plug 등 플러그인 매니저(?)마다 설치할 때의 코드가 다르니, 잘 확인한 후에 설치하도록 합니다. 아래는 제가 설치한 플러그인들입니다.

<br>

### NERDTree

NERDTree는 탐색기 기능을 사용할 수 있는 기능입니다.

```bash
Plug 'scrooloose/nerdtree'
```

 `/`로 파일 검색을 할 수 있으며, `C`는 선택한 폴더를 root 폴더로 만들어줍니다. 네비게이션을 끝내는 명령어는 `q` 입니다. Vim 실행 시 NERDTree를 자동으로 실행시키고 싶다면, 다음을 추가합니다.

```bash
autocmd vimenter * NERDTree
```

위 문구를 추가할 시 NERDTree가 자동으로 실행되긴 하지만, 에디터에서 작업하려면 `:q`를 사용해서 빠져나가야 하므로 불편합니다. 따라서, 위 코드 대신 아래를 사용하여 vim으로 디렉토리를 open 했을 때에만 작동하도록 하는 것이 편리합니다. 개인적으로는 그렇습니다.

```
autocmd StdinReadPre * let s:std_in=1
autocmd VimEnter * if argc() == 1 && isdirectory(argv()[0]) && !exists("s:std_in") | exe 'NERDTree' argv()[0] | wincmd p | ene | exe 'cd '.argv()[0] | endif
```

<br>

### jedi-vim

자동 완성 기능의 플러그인이며, Pydoc이 설치되어 있으면 reference 팝업 기능이 제공됩니다.

```
Plug 'davidhalter/jedi-vim'
```

<br>

### ctags

태그를 등록하여, IDE의 go to definition과 같은 역할을 하도록 합니다. 설치 방법은 OS X와 Ubuntu 모두 쉽지만, OS X의 경우 유의해야 할 것이 있습니다.

```bash
# OS X
$ brew install --HEAD universal-ctags/universal-ctags/universal-ctags

# Ubuntu
$ sudo apt-get install ctags
$ sudo apt install exuberant-ctags
```

만약 `brew install ctags`를 사용할 경우 `Exuberant Ctags`가 설치되므로 [주의해야 합니다(?)](https://johngrib.github.io/wiki/ctags/) ctags에는 Exuberant ctags와 Universal ctags가 있는데, 그 차이는 아직 잘 모르겠습니다.

```bash
# 현재 디렉토리 및 하위 디렉토리까지 모든 클래스, 함수 등을 tags 파일을 생성하여 등록
$ ctags -R .
# 하위 디렉토리까지가 아닌, 현재 디렉토리의 파일들에 대해서만 작업을 진행
$ ctags *

# vim 파일 실행중일 때에도 사용 가능
:!ctags -R
```

참고로, git 사용 시 `tags` 파일을 `.gitignore`에 등록하는 것을 추천합니다.

설정된 태그에 대해 `Ctrl + ]`를 통해 해당 태그로 이동할 수 있으며, `Ctrl + t` 혹은 `:e#` 명령어로 점프 이전 위치로 돌아갈 수 있습니다.

<br>

### ctags not to tag variables nor imported

```bash
$ ctags -R --fields=+l --languages=python --python-kinds=-iv -f /.tags ./
```

`-iv` option은 대상에서 제외하는 기능을 합니다.

<br>

### ctags to All the Python Libraries

가장 어려운 방법은 모두 찾아가서 path를 tag를 생성하는 것입니다. 하지만, 파이썬 코드인 `sys.path`를 활용하여 이를 해결할 수 있습니다.

```bash
$ ctags -R --fields=+l --languages=python --python-kinds=-iv -f ./tags . $(python -c "import os, sys; print(' '.join('{}'.format(d) for d in sys.path if os.path.isdir(d)))")

# 저의 경우, python3라고 명시해야 하므로, 다음과 같이 사용
$ ctags -R --fields=+l --languages=python --python-kinds=-iv -f ./tags . $(python -c "import os, sys; print(' '.join('{}'.format(d) for d in sys.path if os.path.isdir(d)))")
```

위 명령어를 이용한다면, 프로젝트 코드들은 물론이며, std library들과 같은 것들을 모두 태깅하게 됩니다.

<br>

### ctags to Specific Python Library

만약 site-packages에 등록된 라이브러리들 중 특정 라이브러리 만을 태깅하고 싶을 경우, `.vimrc`에 다음과 같이 설정하여 `Ctrl + k` 사용 시 커서의 위치에 존재하는 이름의 라이브러리를 `tags`에 추가하도록 할 수 있습니다.

```bash
# .vimrc
nnoremap <C-k> :!ctags -aR site-packages-경로/<cword><cr>
# 예시
nnoremap <C-k> :!ctags -aR /usr/local/lib/python3.8/dist-packages/<cword><cr>
```

<br>

### easy-plugins

`easy-plugins`를 사용하려면, 우선 `exuberant-ctags`가 설치되어 있어야 합니다. 그리고, vim의 플러그인들 중 하나인 `misc` 또한 필요합니다.

```bash
Plug ''
```



<br>

### Tagbar

현재 파일에서 사용되고 있는 함수, 변수와 같은 내용을 한눈에 볼 수 있도록 해 줍니다.

```
Plug 'majutsushi/tagbar'
nmap <F12> :TagbarToggle<CR>
```

위와 같이 기록해 둔다면, `F12` 키가 Tagbar을 사용하는 단축키가 됩니다. [jen6](https://jen6.tistory.com/119) 이곳에서 Tagbar의 자세한 사용법을 볼 수 있으며, 추가적으로 ctag와 cscope 플러그인을 소개하고 있습니다.

