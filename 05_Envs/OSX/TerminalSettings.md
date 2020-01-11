# Mac Terminal Settings

Default Terminal 대신 사용하는 iTerm2의 설치와 환경설정에 관한 내용입니다.

<br>

<br>

## iTerm2

맥북에서 기본적으로 사용하고 있는 Terminal 대신, 보다 확장된 기능을 수행할 수 있도록 해주는 iTerm2를 설치합니다. Homebrew를 사용하면 보다 편리하게 설치할 수 있습니다.

```bash
$ brew cask install iterm2
```

대/소문자 구분 없이 `Tab` 키를 통해 특정 디렉토리/파일을 보다 쉽게 명령어에 작성할 수 있도록 하는 등의 기능이 있습니다.

<br>

### iTerm Themes

iTerm을 실행한 후 `command+,` 키를 입력하면 설정 창이 나옵니다.

<br>

### Plug-in

Terminal 혹은 iTerm 앱을 보다 편리하게 사용할 수 있도록 도와주는 플러그인입니다.

```bash
# zsh-syntax-highlighting
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# zsh-autosuggestions
git clone git://github.com/zsh-users/zsh-autosuggestions $ZSH_CUSTOM/plugins/zsh-autosuggestions
```

이후 `~/.zshrc` 파일의 `plugins=()`에 설치한 플러그인을 추가합니다.

```bash
# 이미 git은 설치되어 있었다.
plugins=(
  git
  zsh-syntax-highlighting
  zsh-autosuggestions
)
```

`~/.zshrc` 파일을 수정한 후 이를 적용하기 위해 `source ~/.zshrc`를 수행하거나, 터미널을 재시작합니다.