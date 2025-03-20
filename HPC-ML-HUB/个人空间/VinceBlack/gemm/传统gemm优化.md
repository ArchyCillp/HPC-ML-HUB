### 什么是传统
- 不含tensor core
- 不含CC8.0的memcpy async



### 朴素warp乘的问题（如何分析性能瓶颈？）
![](../../../../accessories/Pasted%20image%2020250320192922.png)
-  最常见的两个指标
	- 带宽和延迟
- 分析``c += A[tx * K + i] * B[i * N + ty];``一次循环中
	- 一个warp做32个FFMA（a乘b加c），一个FFMA占用两个OP数，所以64OP
	- 一个warp访问A矩阵1个float，B矩阵的32个float，
