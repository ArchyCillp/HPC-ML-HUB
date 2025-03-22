### 什么是传统
- 不含tensor core
- 不含CC8.0的memcpy async



### 朴素warp乘的问题（如何分析性能瓶颈？）
![](../../../../accessories/Pasted%20image%2020250320192922.png)
-  最常见的两个指标
	- 带宽和延迟
	- 带宽
		- 分析``c += A[tx * K + i] * B[i * N + ty];``
			- 一次循环中，一个warp做32个FFMA（a乘b加c），一个FFMA占用两个OP数，所以64OP
			- 一个warp访问A矩阵1个float，B矩阵的32个float，一共132 bytes
			- 因此`AI = 64 / 132 = 0.48 OP/byte`
			- AI 的使用
				- flops / AI = 需要的内存带宽
				- 内存带宽 x AI = 利用的计算flops
				- AI=0.48很可能无法充分利用计算性能，可以参见[常见显卡算力表](../常见显卡算力表.md)
					- 可以看到一般显卡的做FP32，AI拐点 都在 10～30左右
					- 就算L2 cache全命中，AI拐点也要至少在4以上
	- 延迟
		- 如果Compute和带宽都没有打满，那就可能是没hide latency
		- 主要问题在于指令依赖


### 带宽优化

![](../../../../accessories/Pasted%20image%2020250320201352.png)

- M_block * K_block和K_block * N_block需要能load进shared memory
- 因为shared memory的带宽极大，可以先近似忽略shared memory相对主存访问的代价
- 这样的话对于每个M_block * K_block和K_block * N_block，主存只需要访问一次，现在的AI
![](../../../../accessories/Pasted%20image%2020250320201950.png)
- 通过调整块的大小（比如64x64，AI就可以是16）就可以控制AI，避免memory-bound
- RTX2080FP32FLOPS / 16 = 631.25 GB/s 


### 延迟优化
- 分块的大小会影响延迟
- 分块小了，AI较低
- 分块越大，每个thread被分配的任务越多，可以hide latency（比如ILP）
- 分块太大，寄存器不够，occupancy问题 / K_block通过shared memory也影响occupancy


### thread-level 向量外积 & prefetch
![](../../../../accessories/Pasted%20image%2020250320203405.png)
- 向量外积在循环中有连续的数据复用
	- 寄存器里可以缓存一行B，缓存一个A，做一行计算，然后取下一个A，做一行计算....
	- 这样shared memory的访问数从 M x N x 2 x K -> (M + N) x K
- 这个优化在shared memory带宽是性能瓶颈的时候使用
- prefetch：
	- warp scheduler只有4个，平均指令间隔超过4 cycles，warp scheduler就会大部分时间空闲，无法hide latency
	- 如果latency是主要的问题，还是要尽可能考虑ILP，从数据依赖性考虑prefetch
	- 通过用double buffers，在计算当前的块之前，预先 global -> shared 下一次块的
	- shared也可以preload

