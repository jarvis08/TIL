# Homebrew

- Homebrew 설치하기

  ```shell
  /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
  export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
  brew install python@2  # or python (Python 3)
  ```
  
- Formula 설치 위치

  `/usr/local/Cellar/`

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

<br>

## My Installations

- python3
- docker

