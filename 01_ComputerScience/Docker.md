# Docker

- Check docker works
  - `docker --version`
  - `docker run hello-world`
  - `docker ps --all`
- Look for docker images
  - `docker search ubuntu`
  - `docker pull ubuntu:16.04`
  - `docker images`
- Initialize and run image
  - `docker run -it --name <container_name> ubuntu`
  - `t`는 터미널을 생성한다는 의미
- Run image that was already initialized
  - `docker start <name||id>`
- Access into the docker
  - `docker attach <name||id>`
  - `docker exec -it <name||id> /bin/bash`
    - `exec`: **도커 컨테이너 안쪽에 명령어를 전송할때 사용**
    - `/bin/bash`: 도커 컨테이너 안쪽의 bash 쉘이 실행된다. 접속이란게 결국 리눅스의 쉘을 사용하겠다는 뜻이기 때문에, 이런 방식으로 컨테이너에 접속한다.
- Get out of container
  - `ctrl + p + q`
- Delete specific image
  - `docker rm <container_id>`
- Delete all the images that are not running
  - `docker container prune`

<br>

### Run 명령어에 따른 차이

- `docker run -it`
  - `[Ctrl + P] + [Ctrl + Q]`로 컨테이너에서 빠져나오게 되면 컨테이너를 현재 상태 그대로 두고 외부로 빠져나올 수 있다.
- `docker run -i`
  - `[Ctrl + P] + [Ctrl + Q]`로 컨테이너에서 빠져나올 수 없다. 이 것은 stdin을 붕괴시킬 것이다.
- `docker run`
  - `[Ctrl + P] + [Ctrl + Q]`로 컨테이너에서 빠져나올 수 없다.
  - `SIGKILL` 시그널로 도커 컨테이너를 죽일 수 있다.

`-i` 옵션은 표준 입출력을 사용하겠다는 의미로, `-t` 옵션만 주게되면 `root@134adb2ba12:~/$` 와 같이 프롬프트 모양이 바뀌긴 한다. 하지만 표준 입출력 옵션을 주지 않았으므로 입력을해도 아무런 반응이 없다.

`-t` 옵션은 가상 tty를 통해 접속하겠다는 의미로, `-i` 옵션만 주게되면 `root@134adb2ba12:~/$` 부분이 뜨지 않는다. 그냥 빈 입력화면이 나오는데, 다만 명령어를 입력하면 명령은 제대로 수행된다.