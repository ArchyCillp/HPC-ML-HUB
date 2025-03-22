*由于 double buffer 需要额外的寄存器保存从 global memory 转移到 shared memory 的数据。在安培及其以后的架构下，转移数据并不需要经过寄存器，也就没有这个寄存器开销了，但我是真的没有这张卡*  *--《传统CUDA GEMM不完全指北》*


