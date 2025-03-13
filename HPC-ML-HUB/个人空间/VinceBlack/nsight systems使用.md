## 机器和 setting 要求
必须拥有对 gpu 的 root 权限才能 collect gpu metrics, 因此云gpu基本上不可能使用 nsys. 基本上必须要用自己的机器才行. 
host 是 win, 然后 target 是同一台机子的 wsl 好像也不行, 网上查到的说法是说功能还在开发. 
## 安装注意事项
https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html
- target端执行以下代码
```shell
> cat /proc/sys/kernel/perf_event_paranoid
%if result is larger than 2, run the following% 
> sudo sh -c 'echo 2 >/proc/sys/kernel/perf_event_paranoid'
> sudo sh -c 'echo kernel.perf_event_paranoid=2 > /etc/sysctl.d/local.conf'
```
- 检查可用
```shell
~$ nsys status -e

Sampling Environment Check
Linux Kernel Paranoid Level = 1: OK
Linux Distribution = Ubuntu
Linux Kernel Version = 4.15.0-109-generic: OK
Linux perf_event_open syscall available: OK
Sampling trigger event available: OK
Intel(c) Last Branch Record support: Available
Sampling Environment: OK
```
`nsys status -e` 和 `sudo nsys status -e` 返回的结果不一样的. 
![[../../../accessories/Pasted image 20250302181605.png]]
![[../../../accessories/Pasted image 20250302181611.png]]

## 使用注意事项
Command line with arguments 那边如果 command 比较复杂的话可以写进一个 sh 文件里面去. 然后像下面这样进行 specify:
![[../../../accessories/Pasted image 20250302201443.png]]
![[../../../accessories/Pasted image 20250302201502.png]]

## GUI profiling 

## CUDA Trace
#### nsys提供的功能CUDA trace功能
- 在时间线上追踪CUDA runtime 和 driver API的call
- 在时间线上追踪GPU上发生的事情，比如内存复制，kernel执行
![[Pasted image 20250227172112.png]]
![[../../../accessories/Pasted image 20250227172549.png]]
- memory transfer是红色的，kernel是蓝色的

![[Pasted image 20250227172838.png]]
- 可以调用nsight compute分析具体的kernel


![[Pasted image 20250227173246.png]]
- 可以看到memory usage情况，可以发现memory泄漏的情况（如右图）

(还有CUDA unified memory, page fault, CUDA graph，此处略)


## GPU Hardware Profiling
![](../../../accessories/Pasted%20image%2020250227192412.png)
- nsys还提供对GPU硬件端的profiling，可以回答以下问题：
	- 我的GPU idle了吗？
	- 我的GPU跑满了？grid size和streams够吗？SM和warp slot用满了吗？
	- 我的TensorCore在跑吗？
	- 我的instruction rate高吗？
	- 我是卡在IO或者warp数量上了吗？
- 硬件要求：Turing架构以上

![](../../../accessories/Pasted%20image%2020250227193112.png)
- GUI开启collect GPU metrics
- 值得注意的一些metrics
	- **SMs Active** - `sm__cycles_active.avg.pct_of_peak_sustained_elapsed`
		- 该指标表示在采样周期内，SMs 至少有一个 warp 在执行的时间占比（以百分比表示）。值为 0 表示所有 SMs 都处于空闲状态（没有 warp 在执行）。值为 50% 可能表示以下两种情况之一：
			1. 所有 SMs 在采样周期内有一半的时间处于活动状态
			2. 50% 的 SMs 在整个采样周期内始终处于活动状态
	- **SM Issue** - `sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed`
		- 该指标表示在采样周期内，SM 的子分区（warp 调度器）发出指令的时间占比（以百分比表示）。
		- 因为warp调度器每个cycle都可以发射一个或多个指令，这个指标低表示Compute资源处于空闲状态。
	- **DRAM Bandwidth** `dram__read_throughput.avg.pct_of_peak_sustained_elapsed`
	- **PCIe Throughput** - `pcie__read_bytes.avg.pct_of_peak_sustained_elapsed`
- 注：*`pct_of_peak_sustained_elapsed`指的是峰值持续值，也就是占最大可能值的百分比，比如`sm__cycles_active.avg.pct_of_peak_sustained_elapsed`指的就是SM在工作的cycle数量占全部SM全程在工作的cycle的比例。*