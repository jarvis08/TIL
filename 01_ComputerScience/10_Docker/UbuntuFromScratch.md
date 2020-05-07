# Docker setting with default Ubuntu image

```bash
# pull image of ubuntu
docker pull ubuntu
# use ubuntu image to make 'ubuntuu' container
docker run -it --name ubuntuu ubunt

# use apt to install requirements
apt update && apt upgrade
apt install wget
apt-get install curl

# If Ubuntu 18.04, it will install python3.6
# If Ubuntu 16.04, it will install python3.8
$ apt install libssl-dev libncurses5-dev libsqlite3-dev libreadline-dev libtk8.5 libgdm-dev libdb4o-cil-dev libpcap-dev
$ apt-get install python3-dev

# install gcc
apt install build-essential
apt install cmake

# (if needed, e.g., Ubuntu 16.04) Install python3.6
$ wget https://www.python.org/ftp/python/3.6.9/Python-3.6.9.tgz
tar xvfz Python-3.6.9.tgz
cd Python-3.6.9
./configure
make
make install

# pip 설치의 경우, python3.8에 붙이고 싶으면 3.8 설치 직후에 설치
$ apt install python3-pip
$ apt-get install vim
$ apt install git-all

# pip installations
pip3 install Cython
pip3 install numpy
pip3 install pandas
pip3 install scipy
```

```bash
# vim setting
curl -fLo ~/.vim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

# .vimrc
call plug#begin('~/.vim/plugged')
" Declare the list of plugins.
Plug 'tpope/vim-sensible'
Plug 'junegunn/seoul256.vim'
Plug 'junegunn/vim-easy-align'
Plug 'scrooloose/nerdtree'
Plug 'majutsushi/tagbar'

" Python2
" Plug 'davidhalter/jedi-vim'

" Python3
"Plug 'klen/python-mode'
Plug 'vim-scripts/indentpython.vim'
Plug 'hdima/python-syntax'

"Plug 'sirver/ultisnips'
"let g:UltiSnipsExpandTrigger="<tab>"

" C++
Plug 'octol/vim-cpp-enhanced-highlight'

nmap <F12> :TagbarToggle<CR>
" List ends here. Plugins become visible to Vim after this call.
autocmd StdinReadPre * let s:std_in=1
autocmd VimEnter * if argc() == 1 && isdirectory(argv()[0]) && !exists("s:std_in") | exe 'NERDTree' argv()[0] | wincmd p | ene | exe 'cd '.argv()[0] | endif
call plug#end()

# :source %
# :PlugInstall
```



# 