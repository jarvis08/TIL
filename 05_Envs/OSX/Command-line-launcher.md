# Command-line Launcher

App을 설치한 후, Pycharm과 같은 IDE를 terminal 상에서 실행하고 싶은 경우가 많다. Pycharm의 경우 앱 실행 후 프로젝트를 open 하기가 번거로울 수 있다.

OS X의 경우 symbolic link 혹은 launcher executable file을 만들고, 이를 `/usr/local/bin/launcher-executable-file` 과 같은 형태로 저장하여 터미널에서 사용 가능한 형태로 만들 수 있다. Pycharm의 경우, app 실행 후 menu bar의 `Tools > Create Command-line Launcher` 를 클릭하면 `charm` 이라는 launcher executable file을 생성해 준다. 그리고 생성이 됐다면, `charm` 이라는 명령어로 사용이 가능하다.

이는 `/usr/local/bin`이 default로 `PATH`로 설정되어 있기 때문이며, 아래 명령어를 통해 확인할 수 있다.

```bash
$ cat $PATH
```

