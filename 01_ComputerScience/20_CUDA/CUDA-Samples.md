```bash
# sample 디렉토리
$ cd /usr/local/cuda/samples

# username/group 확인
$ id
user정보 출력

# 권한 부여
$ sudo chown -R <username>:<usergroup> .

# 원하는 sample 이동 후 실행
make
./cu파일명

# nvidia-smi를 1초에 한번씩 출력하도록 하여 실행 확인
$ nvidia-smi -l

# GPU 지정하여 실행 (여기선 0번)
$ CUDA_VISIBLE_DEVICES=0 ./cu파일명
```

