```bash
## 행렬 형태
1 3 5 7
2 4 6 8
# 실제값 = 8
std::cout << data(1, 3) << std::endl
# 실제값 = 7
std::cout << data(0, 3) << std::endl

# 용어 설명
ColMajor: 같은 column의 요소를 다음 순서로 취급
RowMajor: 같은 column의 요소를 다음 순서로 취급
rows_stride: 다른 row의, 다음 row 요소까지의 거리
cols_stride: 다른 col의, 다음 col 요소까지의 거리

## 행렬 저장 형태 1
[1, 2, 3, 4, 5, 6, 7, 8]
num_idx = row * rows_strid() + col * cols_stride()
# ColMajor 검색
rows_stride=1
cols_stride=2
7 = 1 * 1 + 3 * 2 -> # 실제값 = 8
6 = 0 * 1 + 3 * 2 -> # 실제값 = 7
# RowMajor 검색
rows_stride=4
cols_stride=1
7 = 1 * 4 + 3 * 1 -> 실제값 = 8
3 = 0 * 4 + 3 * 1 -> 실제값 = 4

##  행렬 저장 형태 2
# [1, 3, 5, 7, 2, 4, 6, 8]
num_idx = row * rows_strid() + col * cols_stride()
# ColMajor 검색
rows_stride=1
cols_stride=2
5 = 1 * 2 + 3 * 1 -> 실제값 = 4
3 = 0 * 2 + 3 * 1 -> 실제값 = 7
# RowMajor 검색
rows_stride=4
cols_stride=1
7 = 1 * 4 + 3 * 1 -> # 실제값 = 8
3 = 0 * 4 + 3 * 1 -> # 실제값 = 7
```

```cpp
// 예제 2 - nt 계산
// Matrix 표현 형태
// A 배열: [a00, a01, a02, ..., a10, a11, a12, ...]
// B 배열: [b00, b10, b20, ..., b01, b11, b21, ...]
// C 배열: [c00, c01, c02, ..., c10, c11, c12, ...]
// input1에 대한 output = [c00, c01, c02, ...]
// Thus, bias는 row 별로 더해주면 된다.
// A: RowMajor, B: ColMajor, C: RowMajor
int i,j,k;
for(i = 0; i < M; ++i){
  for(j = 0; j < N; ++j){
    register float sum = 0;
    for(k = 0; k < K; ++k){
      sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
    }
    C[i*ldc+j] += sum;
  }
}
```

```cpp
// 예제 2 - nn 계산
int i, j, k;
#pragma omp parallel for
for(i = 0; i < M; ++i){
  for(k = 0; k < K; ++k){
    register float A_PART = ALPHA*A[i*lda+k];
    for(j = 0; j < N; ++j){
      // A: RowMajor, B: RowMajor, C: RowMajor
      C[i*ldc+j] += A_PART*B[k*ldb+j];
    }
  }
}
```

