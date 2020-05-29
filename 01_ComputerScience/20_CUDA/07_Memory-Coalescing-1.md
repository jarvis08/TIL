# Memory Coalescing part-1

Memory Coalescing에 영향을 주는 요인들로 다음을 알아봅니다.

- Misaligned data
- Data Structure
  - Stride
  - Array of structures
  - Structure of arrays

Memory Coalescing은 메모리 단위를 조절하여 효율적으로 read/write가 가능한 형태를 만드는 것입니다. 그러기 위해서는 먼저 메모리가 어떻게 저장되고, 읽히는가를 알아야 합니다. 현재 컴퓨터의 메모리들은 bit 혹은 byte 단위를 사용하여 메모리를 다루는 것이 아니라, **chunk**라는 단위를 사용합니다.

<br><br>

## Chunk

### Main Memory's Chunk Size

우리는 보통 사용할 메모리를 byte 단위로 계산을 하지만, 실제로 main memory의 데이터를 저장할 때 byte 단위가 아니라 **chunk** 단위를 사용합니다. 모든 컴퓨터 구조에서 off-chip memory에 접근할 때에는 chunk 단위를 사용합니다. 이는 single word만을 필요로 할 때에도 마찬가지이며, single word를 읽기 위해서도 해당 word가 포함된 chunk 전체를 읽어야 합니다.

Main memory의 chunk는 과거 32 bytes로 사용됐으며, 현재 사용되고 있는 컴퓨터 구조에서는 64 혹은 128 bytes를 사용합니다. 즉, 우리가 사용하는 main memory는 chunk 단위인 128 bytes를 기준으로 공간을 분할하여 저장합니다.

(그냥 참고: CPU의 경우, 현재 사용중인 64 bit(8 bytes) 체제는 8 bytes를 chunk boundary로 사용합니다)

<br>

### CUDA's Chunk Size

이는 warp 또한 마찬가지입니다. CUDA의 thread들은 32개의 단위로 묶여 warp를 구성하며, 각각의 thread가 float data를 읽으며 총 32개의 float를 read 혹은 write 하게 됩니다. 즉, 4 bytes의 `float`를 thread 별로 하나씩 작업해야 하므로 warp는 총 128 bytes를 다루게 됩니다.

이는 warp size를 왜 더 크게 하지 않는가는 이와 관련이 깊습니다. **Warp가 갖는 thread의 개수를 32개로 정함으로서  메모리로 부터 데이터를 load 하는 단위인 chunk와 동일하게 맞출 수 있고, 따라서 더 효율적으로 데이터를 다루게 됩니다**.

CUDA의 경우 현재 128 bytes를 chunk 단위로 사용하지만, `cudaMalloc()`의 경우 미래의 기술 발전을 고려하여 256 bytes를 boundary로 사용합니다. 따라서 전체 DRAM 공간 중 100 bytes 지점 까지만 사용되고 있다 하더라도, 개발자가 공간 할당 시 101 bytes 지점 부터 할당되는 것이 아니라, 256 bytes 지점의 주소 부터 할당되게 됩니다.
