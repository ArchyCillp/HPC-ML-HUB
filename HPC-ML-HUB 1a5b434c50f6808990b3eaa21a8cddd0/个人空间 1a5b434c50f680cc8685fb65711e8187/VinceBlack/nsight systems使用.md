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
