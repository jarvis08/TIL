# Homebrew

- Homebrew 설치하기

  ```shell
  /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
  export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
  brew install python@2  # or python (Python 3)
  ```
  
- Formula 설치 위치

  `/usr/local/Cellar/`

<br>

## Python3 설치

### 설치 이전

```bash
$  which python3
/usr/bin/python3
$ where python3
/usr/bin/python3
```

macbook의 default python3 경로

<br>

### 설치하기

```shell
$ brew install python3
$ brew rm python3
```

```bash
Python has been installed as
  /usr/local/bin/python3
```

```bash
$ which python3
/usr/local/bin/python3
$ where python3
/usr/local/bin/python3
/usr/bin/python3
```

- `where`: searches for "possibly useful" files
- `which`: only searches for executables.

<br>

### site-packages PATH

`usr/local/lib/python3.y/site-packages/`

<br>

<br>

## Using local git instead of Apple git

[참고 자료](https://www.michaelcrump.net/step-by-step-how-to-update-git/)

최신 버전의 Git을 사용하기 위해, Homebrew를 사용하여 Local Git을 설치합니다. 그리고 기존의 Apple git 대신 Local Git의 PATH를 설정합니다.

```bash
$ brew install git
$ which git
/usr/bin/git
# Once it is installed, then type the following two lines,
# which will set our path to the local git distro instead of the Apple one.
$ export PATH=/usr/local/bin:$PATH
$ git --version
git version 2.24.0
$ which git
/usr/local/bin/git
# You are now updated to the official distro of Git on your Mac.
# To update it in the future all you will need to do
$ brew upgrade git
Error: git 2.24.0_1 already installed
```

<br>

<br>

## oh-my-zsh

`brew`를 사용하여 최신 버전의 zsh과 zsh-completions를 설치합니다.

```bash
$ brew install zsh zsh-completions
```

zsh의 설정을 관리해주는 oh-my-zsh을 설치합니다. 설치 도중 아이디와 패스워드를 입력하면, 자동으로 기본쉘을 bash에서 zsh로 변경합니다.

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
```

<br>

### Plug-in

Oh-my-zsh을 더욱 강력하게 해주는 플러그인을 추가적으로 설치합니다.

```bash
# zsh-syntax-highlighting
$ git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# zsh-autosuggestions
$ git clone git://github.com/zsh-users/zsh-autosuggestions $ZSH_CUSTOM/plugins/zsh-autosuggestions
```

이후 `~/.zshrc`에 플러그인을 설치했음을 알리기 위해 기존 코드를 다음과 같이 변경합니다.

```bash
# ~/.zshrc 변경 전
plugins=(git)
```

```bash
# ~/.zshrc 변경 후
plugins=(
  git
  zsh-syntax-highlighting
  zsh-autosuggestions
)
```

<br>

<br>

## My Installations

- python3
- docker
- git
- zsh
