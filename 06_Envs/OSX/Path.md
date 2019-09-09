# Path

## Path 확인하기

- 현재 설정된 Path 확인하기

  ```shell
  $ echo $PATH
  ```
  
  ```shell
  # 결과
  /usr/local/sbin:/usr/local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/X11/bin:/Applications/Server.app/Contents/ServerRoot/usr/bin:/Applications/Server.app/Contents/ServerRoot/usr/sbin
  ```

- 기본 Path 설정 장소

  `/etc/paths`



## 사용자 PATH


- 저장 파일

  `~/.bash_profile`

### 설정 순서

1. PATH라는 것이 있는지 확인

2. 없을 경우 추가, 공백 주의

   `export PATH=${PATH}` 

그리고 그 뒤에 다른 Path가 필요할 경우 `{PATH}`뒤에 `:` 적어서 경로를 이어준다.

## 관리자 PATH

- `sudo nano /etc/paths`

  ```shell
  # 결과
  /usr/local/bin
  /usr/bin
  /bin
  /usr/sbin
  /sbin
  ```

- 저장 파일

  `/etc/paths`

---