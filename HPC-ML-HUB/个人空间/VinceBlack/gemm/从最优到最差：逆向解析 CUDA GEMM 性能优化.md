### 背景 

最近对着 [CUDA SGEMM矩阵乘法优化笔记——从入门到cublas](https://zhuanlan.zhihu.com/p/518857175) 练习传统gemm，比起做切实的算子，更是想练习一下CUDA调优的流程。

我们都知道cuda调优有很多很多技巧，但是关键是在某种情况下，发现性能瓶颈，实施最有效的优化。

这次，我为了练习gemm，先自己手写了一个naive的gemm，性能是1894gflops。okay，然后我分tile，用shared memory做简单优化(v1)，性能是2471 gflops。

然后进一步，我让每个thread负责的data更多，从1x1改成负责8x8的C子矩阵。然而，性能却退化到了1986 gflops (V2)。

那么问题到底出在哪里？为了不浪费太多时间在写优化和调试上，我网上博客中模仿的一个高性能的版本(V2)出发，一步步进行负优化，通过衰减的性能发现哪些优化对性能的提升最为关键。

结论如下表：

### 性能表

| 版本     | A10性能(GFlops) | thread-level gmem<->smem | unroll    | 向量乘法循环嵌套    | bank conflict相关            | volatile smem | 中间值存储                               | 任务分配                                          |
| ------ | ------------- | ------------------------ | --------- | ----------- | -------------------------- | ------------- | ----------------------------------- | --------------------------------------------- |
| cublas | 13872         | -                        | -         | -           | -                          | -             | -                                   |                                               |
| V2     | 9784          | 以float4为单位读写             | TRUE      | k->m->n     | 无特殊处理                      | FALSE         | float r_c[8][8]                     | standard tiling                               |
| V1.9   | 8606          | **4个连续单独的float读写**       | TRUE      | k->m->n     | 无特殊处理                      | FALSE         | float r_c[8][8]                     | standard tiling                               |
| V1.8   | 8388          | 4个连续单独的float读写           | **FALSE** | k->m->n     | 无特殊处理                      | FALSE         | float r_c[8][8]                     | standard tiling                               |
| V1.7   | 6056          | **单个float读写**            | FALSE     | k->m->n     | 无特殊处理                      | FALSE         | float r_c[8][8]                     | standard tiling                               |
| V1.7.1 | 1057          | 单个float读写                | FALSE     | k->m->n     | 无特殊处理                      | **TRUE**      | float r_c[8][8]                     | standard tiling                               |
| V1.7.2 | 6959          | 单个float读写                | **TRUE**  | k->m->n     | 无特殊处理                      | FALSE         | float r_c[8][8]                     | standard tiling                               |
| V1.6   | 1931          | 单个float读写                | FALSE     | **m->n->k** | 无特殊处理                      | FALSE         | float r_c[8][8], 实际发现在local memory里 | standard tiling                               |
| V1.6.1 | 1947          | 单个float读写                | FALSE     | m->n->k     | **smem每行加16bytes padding** | FALSE         | float r_c[8][8], 实际发现在local memory里 | standard tiling                               |
| V1.6.3 | 990           | 单个float读写                | FALSE     | m->n->k     | 无特殊处理                      | **TRUE**      | float r_c[8][8], 实际发现在local memory里 | standard tiling                               |
| V1.6.4 | 5090          | 单个float读写                | FALSE     | m->n->k     | 无特殊处理                      | FALSE         | **`__shared__ float r_c[8][8]`**    | standard tiling                               |
| V1     | 1986          | 单个float读写                | FALSE     | m->n->k     | 无特殊处理                      | FALSE         | float r_c[8][8], 实际发现在local memory里 | tiling, 8x8 data/thread, 16x16 blockDim, K=8  |
| V0     | 2471          | 单个float读写                | FALSE     | -           | 无特殊处理                      | FALSE         | -                                   | tiling, 1x1 data/thread, 16x16 blockDim, K=16 |
| naive  | 1894          | 单个float读写                | FALSE     | -           | -                          | -             | -                                   | 每个thread负责一个C元素 (1x1 data/thread)， 无smem使用    |

### 基本分析
- V2->V1.9, global月shared的数据交换中，把一个float4拆成4个float的指令，有900的衰退，说明指令数量确实有一定影响，应该主要影响的是memory busy、内存访问合并压力。
	- 需要进一步ncu验证
- V1.9->V1.8，unroll，有一定小影响，300左右。
- V1.8->V1.7，因为每个block有256个thread，K每次迭代中需要load A和B的tile（每个1024个float），所以每个thread需要load4个float，V1.8之前都是thread负责4个相邻的float，一共而V1.7改成block总体一轮load256个相邻的float（用四轮），性能下降了2300。
	- V1.8写shared memory，每个thread用2个STS.128指令写A和B的tile
	- V1.7写shared memory，每个thread用8个STS指令写A和B的tile
	- 更重要的是，这似乎竟然能影响之后计算AxB的部分，虽然两处的CUDA代码是一样的，但是编译器做出了不同的决断。貌似V1.8在读shared memory的时候也选择全用LDS.128去读，而V1.7就不是，有很多LDS。这导致V1.7使用了更多的shared memory读写指令，从而在DRAM方面基本持平的情况下，因为指令数量问题造成了性能劣势。
	- 因此，用float4与shared memory进行交互还是很重要的。
- V1.7->V1.6，性能大跳水，只改了一个向量乘法的嵌套顺序。
	- 经过V1.6.x的多方面验证，发现主要问题是出现了不该出现的local memory。
	- 哪来的local memory呢？发现是存线程中间结果的`float r_c[8][8]`数组竟然跑到local memory去了。
	- 为什么之前r_c在寄存器里，改了个乘法嵌套顺序就变了？发现可能是如下原因：
		- register的数量虽然是256个/thread，但是为了occupancy，编译器可能只会给你尽可能优化到用128个，这样你就跑2blocks/SM了。
		- r_c要占用64个寄存器，而O3的编译器老想着缓存。
		- V1.8之后的版本里，k在最外层（向量外积），所以内层的m和n循环一共才需要一行B和一列A（16个float），缓存压力较小。
		- 而V1.7里，k在最里层（向量内积），这导致A和B需要被重复访问的次数大大增加。编译器因此选择用了更多的寄存器去缓存这里，从而导致r_c因为占用寄存器太多被扔到local memory里。（事实证明编译器做了个很差的选择）
		- 这一点从ptxinfo编译时的输出`ptxas info : Used 92 registers, used 1 barriers, 256 bytes cumulative stack size, 8192 bytes smem, 396 bytes cmem[0]`中的cumulative stack size可以看到，有趣的是这并不会在spill里体现出来。
		- 另外在ncu的source也可以注意到local memory的读写，以及激增的对global memory的读写（本质是一个东西）。
		- V1.6.4里，我们选择把r_c扔到shared里，发现性能又回来了不少，进一步验证了这一点。
	- 至于怎么解决这个问题，目前没找到强制数组进register的好方法，建议关注ptxinfo的输出，以及ncu的时候注意是否有local memory的读写发生。实在不行就把shared memory临时做寄存器使用。