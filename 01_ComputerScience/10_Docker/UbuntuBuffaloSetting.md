# How to Set Ubuntu for Buffalo

```bash
# use ubuntu bionic image
$ docker run -it --name Ubuntu18.04 c3c304cb4f22
$ docker exec -it Ubuntu18.04 /bin/bash

$ apt update && apt upgrade
$ apt install wget
$ apt-get install curl

# install python3.6
$ apt install libssl-dev libncurses5-dev libsqlite3-dev libreadline-dev libtk8.5 libgdm-dev libdb4o-cil-dev libpcap-dev
$ apt-get install python3-dev

$ apt install build-essential
$ apt install cmake
$ apt-get install vim
$ apt install git-all

# must upgrade essential tools
$ apt install python3-pip
$ pip3 install --upgrade pip
$ pip3 install --upgrade setuptools
$ pip3 install --upgrade wheel
$ pip3 install Cython
$ pip3 install numpy

# install buffalo from source
$ cd /tmp
$ git clone -b master https://github.com/kakao/buffalo
$ cd buffalo
$ git submodule update --init
$ pip3 install -r requirements.txt
$ python3 setup.py install

# test buffalo, for me, got bunch of errors
$ cd /tmp/buffalo/tests
$ pytest ./algo/test_als.py -v
# I got bunch of errors, though didn't get any error about Cython
```

TensorFlow와 Buffalo 모두 정상 동작 하는 것을 확인했다. Buffalo의 경우 Evaluate을 init하는 과정에서 Cython의 so 파일을 실행하는것이 가장 큰 문제였었다. 문제가 발생했던 것은 Docker를 사용한 Ubuntu16.04 > Python3.6/3.7/3.8 환경들이었다. 실패했던 버전들은 pip install buffalo와 source 설치를 모두 진행했었다. 시스템 환경에도, 가상 환경에도 설치해 보았었다.

그 후 Ubuntu18.04가 default로 Python3.6을 사용한다는 것을 알았고, 18.04 버전의 Docker image를 이용하여 Python3.6을 걸치했더니 문제가 해결됐다. 그런데 신기한 것은, 이전 실패한 환경에서는 pytest 진행 시 cython 문제를 제외하면 모두 pass 했다.. Cython 에러의 내용은 다음과 같다.

```bash
_core.cpython-37m-x86_64-linux-gnu.so: undefined symbol:
```

