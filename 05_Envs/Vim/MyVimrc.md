# My ./vimrc Setting

```bash
call plug#begin('~/.vim/plugged')
""" Common Plug-in
Plug 'tpope/vim-sensible'
Plug 'junegunn/seoul256.vim'
Plug 'junegunn/vim-easy-align'
Plug 'majutsushi/tagbar'
nmap <F12> :TagbarToggle<CR>
Plug 'scrooloose/nerdtree'
autocmd StdinReadPre * let s:std_in=1
autocmd VimEnter * if argc() == 1 && isdirectory(argv()[0]) && !exists("s:std_in") | exe 'NERDTree' argv()[0] | wincmd p | ene | exe 'cd '.argv()[0] | endif

""" About ctags
"Plug 'xolox/vim-misc'
"Plug 'xolox/vim-easytags'
""" ctags configures
set tag=./tags;/
"" <Ctrl + k> key to add the library to tags
nnoremap <C-k> :!ctags -aR /usr/local/lib/python3.8/dist-packages/<cword><cr>
"" async loading tags
"let g:easytags_async=1
"" tag highlight off
"let g:easytags_auto_highlight = 0
"" also find member of structure
"let g:easytags_include_members = 1

""" Python2
"Plug 'davidhalter/jedi-vim'

""" Python3
"Plug 'klen/python-mode'
Plug 'vim-scripts/indentpython.vim'
Plug 'hdima/python-syntax'
"Plug 'sirver/ultisnips'
"let g:UltiSnipsExpandTrigger="<tab>"

""" C++
Plug 'octol/vim-cpp-enhanced-highlight'
call plug#end()

""" Default Vim Settings
augroup CursorLine
    au!
    au VimEnter,WinEnter,BufWinEnter * setlocal cursorline
    au WinLeave * setlocal nocursorline
augroup END

noremap <Up> <Nop>
noremap <Right> <Nop>
noremap <Left> <Nop>
noremap <Down> <Nop>
```

