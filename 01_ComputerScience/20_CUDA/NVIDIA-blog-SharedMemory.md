# Shared Memory

### Shared Memory란

- Shared memory는 on-chip이기 때문에 global memory 보다 훨씬 빠르다.
  - Latency는 uncahced global mamory에 비해 대략 100배 작다.
- **Thread block 마다 하나 shared memory를 보유**
  - Thread block 내의 thread들 중 하나가 global memory로 부터 shared memory로 데이터를 load
  - 같은 thread block 내의 thread들은 같은 shared memory를 공유

<br>

## Shared Memory 사용하기

```cpp
__global__ void staticReverse(int *d, int n)
{
  __shared__ int s[64];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

__global__ void dynamicReverse(int *d, int n)
{
  // extern: 외부 참조 변수/함수
  extern __shared__ int s[];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

int main()
{
  const int n = 64;
  int a[n], r[n], d[n];

  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1;
    d[i] = 0;
  }

  int *d_d;
  cudaMalloc(&d_d, n * sizeof(int)); 
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  
  // run version with static shared memory
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  staticReverse<<<1,n>>>(d_d, n);
  cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);

  // run dynamic shared memory version
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  dynamicReverse<<<1,n,n*sizeof(int)>>>(d_d, n);
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);
}
```

위 코드는 shared memory를 사용하여 64개 요소를 가진 배열을 거꾸로 뒤집는 작업을 구현한  내용입니다. 두 커널 함수는 동일한 역할을 하지만, 차이점이 두 가지가 있습니다.

- Shared memory 내부 배열들의 선언 방법
- Kernel들의 호출 방법

<br>

### Thread Synchronization

Block 내의 모든 thread 들이 공유하는 만큼, shared memory를 사용할 때에는 race condition들에 주의해야 합니다. 모든 thread들이 logically parallel하게 실행되지만 물리적으로 동시에 실행되는 것은 아니기 때문입니다.

Thread A와 B가 서로 다른 데이터를 global memory로부터 load 했으며, A와 B는 서로 다른 warp에 속해 있습니다. 그런데 B가 데이터를 shared memory에 쓰는 작업을 완료하기 전에 A가 그 데이터를 읽으려 한다면, 이 때 race condition이 존재하게 되며, undefined behavior가 발생하게 되고, incorrect result를 얻게 됩니다.

이 때 우리는 thread synchronization이 필요한 것이며, 이는 `__synchthreads()`로 수행할 수 있습니다. `__synchthreads()` 코드는 이전에 호출된 thread 작업이 종료되기 전에는 다음 thread의 작업을 시작하지 않도록 합니다. 그런데 `__synchthreads()`를 잘 못 사용하게 되면 서로 다른 thread들이 상대 작업이 끝나기를 기다리기만 하는 **dead lock**에 빠질 수 있기 때문에 주의해야 합니다.

```cpp
__global__ void staticReverse(int *d, int n)
{
  __shared__ int s[64];
  int t = threadIdx.x;
  int tr = n-t-1;
  // 이전 thread 작업
  // global memory의 d[t]를 shared memory s[t]로 load
  s[t] = d[t];
  __syncthreads();
  // 다음 thread 작업
  // shared memory의 s[tr]을 global memory d[t]로 push
  d[t] = s[tr];
}

__global__ void dynamicReverse(int *d, int n)
{
  extern __shared__ int s[];
  int t = threadIdx.x;
  int tr = n-t-1;
  // 이전 thread 작업
  s[t] = d[t];
  __syncthreads();
  // 다음 thread 작업
  d[t] = s[tr];
}
```

