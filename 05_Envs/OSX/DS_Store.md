# DS_Store

OS X에서는 자동으로 DS_Store라는 directory 명세 파일이 생성됩니다. 따라서 이를 .gitignore에 추가해주지 않는다면, 디렉토리 마다 이 파일이 존재한 상태로 `push`됩니다. 이를 방지하여, 자동으로 모든 DS_Store를 .gitignore에 추가되는 작업을 해 보겠습니다.

```bash
$ echo ".DS_Store" >> ~/.gitignore_global
$ echo "._.DS_Store" >> ~/.gitignore_global
$ echo "**/.DS_Store" >> ~/.gitignore_global
$ echo "**/._.DS_Store" >> ~/.gitignore_global
$ git config --global core.excludesfile ~/.gitignore_global
```

위와 같이 echo 명령어를 사용할 수 있으며, 직접 다음과 같이 작성하여 넣어줄 수 있습니다.

```bash
$ touch .gitignore_global
$ vi .gitignore_global
# 아래 네 줄을 입력한 후, :wq를 사용하여 vi를 종료
.DS_Store
._.DS_Store
**/.DS_Store
**/._.DS_Store
$ git config --global core.excludesfile ~/.gitignore_global
```

