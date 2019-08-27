# Virtual Environment

---

```shell
mkdir python-verualenv
# -m : modul
python -m venv ~/설치경로/원하는_가상환경_폴더명
# python3 -m venv ~/Documents/99_venv/3.7.4
cd work-directory
# 가상환경 실행
source 설치경로/원하는_가상환경_폴더명/Scripts/activate
# mac os
source 설치경로/원하는_가상환경_폴더명/bin/activate
# 가상환경 종료
deactivate

# alias 이용하여 실행 코드 줄이기
# .bashrc 추가, mac OS의 경우 ~/.bash_profile
alias venv='source ~/python-virtualenv/3.7.3/Scripts/activate'
source .bashrc
source .bash_profile
```

