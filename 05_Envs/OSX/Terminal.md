# Terminal

```bash
$ cat ~/파일
파일 터미널에 읽어줌
$ echo '내용' >> 파일명
내용을 파일명에 추가 기록
```

<br>

<br>

## Process Info

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

### 시간 순서로 정렬하여 보기

```bash
$ ls -ltr
```

