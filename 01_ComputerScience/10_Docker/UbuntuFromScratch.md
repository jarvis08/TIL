# Docker setting with default Ubuntu image

```bash
# pull image of ubuntu
docker pull ubuntu
# use ubuntu image to make 'ubuntu' container
docker run -it --name ubuntu ubunt

# use apt to install requirements
apt update && apt upgrade
apt install wget
apt-get install curl
apt-get install vim
# command below installs python3.8 & pip of it
apt-get install python3-dev
apt install python3-pip
apt install git-all
# install gcc and other things
apt install build-essential
apt install cmake

# pip installations
pip3 install Cython
pip3 install numpy
pip3 install pandas
pip3 install scipy
```

```bash
# vim setting
curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
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



