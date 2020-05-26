# CUDA Programming

[NVIDIA Documentation](https://devblogs.nvidia.com/even-easier-introduction-cuda/)을 통해 공부한 내용입니다.

### Initialization

```cpp
int main()
{
    const unsigned int N = 1048576;
    const unsigned int bytes = N * sizeof(int);
    int *h_a = (int*)malloc(bytes);
    int *d_a;
	// d_a의 주소(&d_a)가 가리키는 주소(* &d_a)에 bytes 만큼의 값(** &d_a)을 할당
    cudaMalloc((int**)&d_a, bytes);

    memset(h_a, 0, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);

    return 0;
}
```

<br>

## Single thread GPU calculation

일반적인 cpp 코드로 두 배열의 요소들을 더하는 `add()` 함수를 제작합니다. 이 함수를 GPU를 사용하여 사용하는 방법은 간단합니다.

GPU에서 함수는 `kernel` 이라고 표현합니다.

```cpp
#include <iostream>
#include <math.h>

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void add(int n, float *x, float *y)
{
  // use 1 block
  //int index = threadIdx.x;
  //int stride = blockDim.x;

  // use blocks as much as needed
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x; // gird 내의 전체 thread 개수

  //for (int i = 0; i < n; i++)
  for (int i = index; i < n; i += stride)
      // index부터 stride까지 모든 thread에 값을 할당 후 계산
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elements

  //float *x = new float[N];
  //float *y = new float[N];
  
  // Allocate Unified Memory -- accessible from CPU or GPU
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // cudaMallocManaged()는 unified memory를 사용하여 H2D, D2H의 데이터 송수신을 없앤 것
  // cudaMalloc()을 사용한 분리된 메모리 공간 사용은 속도가 빠른 것이 장점
  // cudaMalloc((float**)&y, N*sizeof(float))

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  // unified memory가 아닐 경우 memset(x, 0.0, N*sizeof(float))

  //add(N, x, y);
  // Run kernel on 1M elements on the CPU

  // use 1 thread
  // add<<<1, 1>>>(N, x, y);

  // use 1 block
  add<<<1, 256>>>(N, x, y);

  // use blocks as much as needed
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize; // int는 버림을 이용하므로
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  //delete [] x;
  //delete [] y;
  cudaFree(x);
  cudaFree(y);

  return 0;
}
```

- `__global__` : 함수가 GPU 상에서 작동함을 CUDA C++ compiler에게 선언하며, CPU 코드로 부터 호출됨
- `cudaMallocManaged(&array_name, num_elements * sizeof(type_name))` : C, C++의 `malloc`과 같은 기능을 하며, __Unified Memory__에 데이터를 할당
pointer를 반환하며, 해당 포인터는 hist(CPU)와 device(GPU) 모두에서 사용 가능
- `cudaFree(array_name)` : free the data allocated before
- `kernel_name<<<NumOfBlocksToUse, NumOfThreadsPerBlock>>>(params, of, the, kernel)` : `<<< >>>`은 GPU 상에서 해당 kernel을 실행
	- `threadIdx.x ` : 현재 block, block의 thread들 중 현재 thread의 index
	- `blockDim.x` : block의 thread 개수
	- SM, Streaming Multiprocessor : CUDA GPU들의 parallel processor들은 SM이라는 단위로 grouped
	각 SM들은 multiple concurrent thread block들을 실행 가능
- Grid : `NumOfBlocksToUse`에 따라 grid를 생성하게 되는데, 이는 사용하게 될 모든 thread를 단위화
	- `gridDim.x` : grid에 포함된 block의 개수
	- `grid 내의, 특정 block의, 특정 thread의 index = `blockIdx.x * blockDim.x + threadIdx.x`
- `cudaDeviceSynchronize()` : CUDA kernel이 실행될 때 자동으로 CPU thread를 막지는 않으므로, GPU 계산 결과를 기다린 후 나머지 CPU 코드를 실행하기 위해 계산 종료를 기다리도록 명령

CUDA 파일들은 `nvcc`를 이용하여 compile하며, `nvcc`는 `.cu` 확장자의 파일만을 다루므로 `add.cu`라고 파일명을 설정합니다.

`$ nvcc add.cu -o add_cuda`

위 명령어를 사용하여 compile할 수 있으며, compile 후 `$ ./add_cuda`를 입력하여 실행합니다.

`$ nvprof ./add_cuda`를 명령하면 profiling을 할 수 있습니다.

```bash
Error: unified memory profiling failed.
# nvprof 후 위와 같은 에러 발생 시, 다음과 같이 명령
$ nvprof --unified-memory-profiling off ./add_cuda
```
