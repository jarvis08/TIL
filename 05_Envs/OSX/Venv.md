# Virtual Environment

## Python 2

1. Homebrew를 사용하여 Python 2 설치합니다.

   ```bash
   $ brew install python@2
   $ which python
   /usr/bin/python
   
   export PATH="/usr/local/opt/python@2/libexec/bin:$PATH"
   $ which python
   /usr/local/bin/python
   ```

   _Homebrew names the executable `python2` so that you can still run the system Python via the executable `python`._

2. virtualenv를 설치합니다.

   `pip install virtualenv`

3. 원하는 디렉토리로 이동 후, 이름(py2)과 버전을 설정하여 가상환경을 설치합니다.

   `virtualenv py2 --python==python2.7`

4. 가상환경을 실행합니다.

   ```bash
   source ~/path/to/venv/py2/bin/activate
   ```

5. 보다 편리하게 가상환경을 실행할 수 있도록, `.zshrc`에 `alias`를 설정합니다.

   ```bash
   alias py2="source ~/Documents/09_Venv/py2/bin/activate"
   ```

   

   

