# Virtual Environment

---

```shell
# -m : modul
python -m venv ~/설치경로/원하는_가상환경_폴더명
source 설치경로/Scripts/activate

# OS X
python3 -m venv ~/Documents/99_venv/3.7.4
source 설치경로/bin/activate

# 가상환경 종료
deactivate

# alias 이용하여 실행 코드 줄이기
# .bashrc 추가, mac OS의 경우 ~/.bash_profile
alias venv='source ~/python-virtualenv/3.7.3/Scripts/activate'
source .bashrc
source .bash_profile
```

```bash
# python 2
$ python -m virtualenv venv
$ virtualenv venv --python=python
$ virtualenv venv --python=python2.7

# python 3
$ python3 -m virtualenv venv
$ virtualenv venv --python=python3
$ virtualenv venv --python=python3.5
```

