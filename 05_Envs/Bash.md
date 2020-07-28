# Terminal

```bash
$ cat ~/파일
파일 터미널에 읽어줌
$ echo '내용' >> 파일명
내용을 파일명에 추가 기록
```

```bash
# <대상> 문자열이 들어간 내용만 출력
$ <행위>|grep <대상>
# banana라는 단어가 들어간 것들로만 출력
$ cat fruits.txt|grep banana

# <대상> 문자열이 안들어간 내용만 출력
$ <행위>|grep -v <대상>
# banana라는 단어 제외하고 검색
$ cat fruits.txt|grep -v banana
# banana, apple, kiwi 제외하고 검색
$ cat fruits.txt|grep -v banana|grep -v apple|grep -v kiwi

# <대상|대상|대상|...> 문자열들이 안들어간 내용만 출력
$ <행위>|grep -ev <대상|대상|대상|...>
# banana, apple, kiwi 제외하고 검색
$ cat fruits.txt|grep -ev banana|apple|kiwi
```

```bash
# 시간 순서로 정렬 및 디렉토리 내용 출력
$ ls -ltr
```

<br><br>

## Process

### Process Info

실행 중인 process는 `ps -efjc`를 통해 조회할 수 있으며, and를 의미하는 |`와 함께  `grep [process_name]`을 사용하면 대상을 특정지을 수 있습니다.

```bash
# 실행중인 process 목록 조회
$ ps -efjc
# 실행중인 process 목록 중 monogs를 grep
$ ps -efjc|grep mongos
```

<br>

### Background Process

Background에서 실행되는 process는 `-efjc` 대신 `aux`를 사용해야 합니다.

```bash
# 실행중인 background process 목록 조회
$ ps aux
# 실행중인 background process 목록 중 supervisor를 grep
$ ps aux|grep supervisor
```

<br>

### Example

오류가 발생한 supervisor의 pid를 검색하여 강제종료 시키기

```bash
$ ps aux|grep supervisor
# 결과: supervisor의 정보와 grep 정보 두 개가 출력
kakao             5697   0.0  0.1  4285496  19988 s000  S+    3:29PM   0:00.20 /usr/local/Cellar/supervisor/4.1.0_1/libexec/bin/python3.8 /usr/local/bin/supervisorctl -s http://localhost:23231
kakao             5671   0.0  0.0  4304700   7280   ??  Ss    3:28PM   0:24.31 /usr/local/Cellar/supervisor/4.1.0_1/libexec/bin/python3.8 /usr/local/bin/supervisord -c spvc1.conf
kakao             7724   0.0  0.0  4268296    676 s001  S+    1:33PM   0:00.00 grep --color=auto --exclude-dir=.bzr --exclude-dir=CVS --exclude-dir=.git --exclude-dir=.hg --exclude-dir=.svn supervisor
# supervisor의 pid인 5697을 강제 종료
$ kill -9 5697
```

<br>

<br>

## Resource Management

```bash
$ htop
$ nvidia-smi
```

### DRAM 확인

```bash
$ free
# 1초 단위로 확인(-g는 기가바이트 단위로 보겠다는 의미)
$ free -s 1
$ watch -n 1 free -g
```

### Disk 용량 확인

```bash
$ df
# G, MB 등의 단위로
$ df -h
```

### File System

```bash
# tmp 확장자를 가진 파일 모두 삭제
# 하위 디렉토리까지 적용
$ find . -name '*.tmp' -exec rm {} \;
```

```bash
# 하위 디렉토리로부터 cost = 이라는 단어를 포함하는 파일을 탐색
# md가 들어가는 파일은 제외
$ grep -R "cost =" * |grep -v md
```

```bash
# L1이 들어가는 파일을 cfg 디렉토리 내에서 탐색
$ grep -R L1 cfg/*
```







