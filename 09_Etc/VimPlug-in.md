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

vim 실행 시 NERDTree를 자동으로 실행시키고 싶다면, 다음을 추가합니다.

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

## Tagbar

IDE들에 흔히 존재하는 `Go to definition` 기능을 해줍니다.

```
Plug 'majutsushi/tagbar'
nmap <F12> :TagbarToggle<CR>
```

위와 같이 기록해 둔다면, `F12` 키가 Tagbar을 사용하는 단축키가 됩니다. [jen6](https://jen6.tistory.com/119) 이곳에서 Tagbar의 자세한 사용법을 볼 수 있으며, 추가적으로 ctag와 cscope 플러그인을 소개하고 있습니다.

