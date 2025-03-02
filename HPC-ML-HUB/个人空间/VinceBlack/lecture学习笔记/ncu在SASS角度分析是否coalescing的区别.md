SASS指令解释参考 cloudcore 的知乎：https://zhuanlan.zhihu.com/p/163865260
SASS取模的实现： https://forums.developer.nvidia.com/t/how-does-this-logic-for-the-modulo-operation-implementation-work/277485

## 两段代码
```C++
//NCcopy
__global__ void copyDataNonCoalesced(float *in, float *out, int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n) {
		out[index] = in[(index * 2) % n];
	}
}

//Ccopy
__global__ void copyDataCoalesced(float *in, float *out, int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n) {
		out[index] = in[index];
	}
}
```

两段代码，都是用了128的block来直接复制长度16M的float数组，区别就是CCopy是每个warp内的thread复制的是连续的32个float，而NCCopy的每个warp复制的是中间隔着一个的32个float。

根据常规的CUDA对于memory coalescing的优化策略我们可以知道，NCCopy的方式coalescing较差，应该是会慢，但是实际上是否coalescing在nsight compute中体现出的指标会是怎样的呢？我们实际来看一下。

以下是两段代码的一些ncu指标对比：

|                                              | NCCopy | CCopy  | 比例   |
| -------------------------------------------- | ------ | ------ | ---- |
| Time (us)                                    | 405.92 | 270.69 | 1.5  |
| Warp Cycles Per Executed Instruction (cycle) | 52.98  | 86.79  |      |
| Executed Instruction (M inst.)               | 18.35  | 6.82   | 2.69 |
| Achieved Occupancy (%)                       | 81.61  | 74.19  |      |
| CPI x EI / AO                                | 11.91  | 7.97   | 1.49 |

可以看到，这CPI，EI和AO三个指标对时间的贡献关系是直接成立的。那么，我们就按照常规经验，分别分析这三个方面的因素对时间的影响。其中，NCCopy变慢的最重要的原因就是它执行的指令（也就是SASS指令）数量是CCopy的2.69倍。但是，为什么几乎同样的CUDA函数，只是写global的位置不同，就会差这么多指令数量呢？而且，为什么NCCopy的CPI会快呢，这很反直觉，不是colasecing会快吗？让我们一探究竟。

## SASS指令对比

#### CCopy
![](../../../../accessories/Pasted%20image%2020250302023637.png)
首先看CCopy的SASS指令，我们来逐句分析
-  `MOV R1, c[0x0][0x28]`
	- 从常量内存(也就是c)导入到寄存器R1，作用未知，后续未使用R1
- `S2R R4, SR_CTAID.X` 
	- 把block ID导入到R4
- `S2R R3, SR_TID.X`
	- 把thread ID导入到R3
- `IMAD R4, R4, c[0x0][0x0], R3` 
	- 计算R4 乘 blockNum（存在常量区里）加R3，最终结果写到R4，也就是index了
-  `ISETP.GE.AND P0, PT, R4, c[0x0][0x170], PT` , `@P0 EXIT`
	- P0 = R4 >= c0x170(也就是n)，如果P0是true的话，结束程序
到这里为止都很简单，R4是thread全局ID，接下来就是关键了
- `MOV R5, 0x4`
	- R5=4，也就是float的大小
- `ULDC.64 UR4, c[0x0][0x118]`
	- 导入一个常量到UR4（uniform寄存器，是GPU中warp级别共享的寄存器），后续没有显式的被用到，猜测可能是跟参数里的in和out的地址偏移相关；ncu说后面LDG和STG都用到了UR4和UR5（64位所以占两个），可能是隐式的使用了
- `IMAD.WIDE R2, R4, R5, c[0x0][0x160]`
	- R2(64位即R2,R3) = R4(即index) x R5(即sizeof(float)) + 常量(in的地址偏移)
- `LDG.E R3, [R2.64]`
	- 读global的R2位置的数据到R3，也就是in\[index\]
	- R3 <=global= * R2
- `IMAD.WIDE R4, R4, R5, c[0x0][0x168]`
	- R4(64位即R4,R5) = R4(即index) x R5(即sizeof(float)) + 常量(out的地址偏移)
- `STG.E [R4.64], R3`
	- * R4 <=global= R3
至此我们可以看到，CCopy在`out[index]=in[index]`上也就是六七句SASS指令。
#### NCCopy

![](../../../../accessories/Pasted%20image%2020250302035527.png)
相比之下，NCCopy的指令非常之长。
其中，开头双方相同的部分都是6条指令。而双方不同的地方在于，复制命令那句CCopy只有7条指令，而NCCopy有29条。这就是导致为什么NCCopy的指令执行数量是CCopy的2.69倍的原因。让我们看一下它到底干了什么：

- `IABS R7, c[0x0][0x170]`
	- R7 = |n|
-  `IMAD.SHL.U32 R0, R4, 0x2, RZ`
	- R0=R4(即index) x 2 + 0
	- R0即 index x 2
	- 因为乘数是2，所以用了SHL.U32直接做移位
- `ULDC.64 UR4, c[0x0][0x118]`
	- 跟CCopy一样的
-  `I2F.RP R5, R7`
	- R5 = float(R5)
	- *为什么要给n转成float？*
	- 我查到了，这一系列看起来奇怪的操作都是用了一个算法算取模，参考开头的引用链接
	- 牛顿迭代法算除法和取模。
-  `ISETP.GE.AND P2, PT, R0, RZ, PT`
	- if (R0 >= 0): P2 = true
	- 也就是if (index x 2 >= 0): P2 = true
	- *比较迷惑，感觉是做模的一系列判断*
- `MUFU.RCP R5, R5`
	- R5 = 1 / R5
	- R5 成了 n的倒数
- `IADD3 R2, R5, 0xffffffe, RZ`
	- R2 = R5(即1/n) + 0xffffffe + 0
	- 给R5减去2，说是为了牛顿迭代里面让近似值调整为低估值
- `IMAD.MOV.U32 R5, RZ, RZ, 0x4`
	- 等同于 R5 = 4
	- 根据cloudcore的讲解，使用IMAD.MOV.U32的原因，是因为MOV用的是整数ALU的发射端口，而IMAD用的是浮点数ALU的发射端口，这样写可以利用硬件资源，和整数ALU的指令发射错开
-  F2I.FTZ.U32.TRUNC.NTZ R3, R2
	- R3 = int(R2)，后面的一堆修饰作用如下
```C++
float x = R2;          // 从 R2 读取浮点数值
if (is_denormal(x)) {  // 如果 x 是非正规数
    x = 0.0f;          // 将其视为0（FTZ 的作用）
}
unsigned int result = (unsigned int)truncf(x);  // 截断并转换为无符号整数（TRUNC 和 U32 的作用）
R3 = result;           // 将结果存储到 R3
```

- `MOV R2, RZ`
	- R2=0
- `IADD3 R6, RZ, -R3, RZ`
	- R6 = -R3
- `IMAD R9, R6, R7, RZ`
	- R9 = R6 x R7
	- 也就是R9 = -1/n的低估值 x n
	- 这是牛顿迭代法的其中一步
- IABS R6, R0
	- R6 = |R0| (index)
- IMAD.HI.U32 R3, R3, R9, R2
	- `R3 = ((R3 * R9) >> 32) + R2;`
- IMAD.HI.U32 R3, R3, R6, RZ
IMAD.MOV R3, RZ, RZ, -R3
IMAD R2, R7, R3, R6
ISETP.GT.U32.AND P0, PT, R7, R2, PT
@!P0  IADD3 R2, R2, -R7, RZ
ISETP.NE.AND P0, PT, RZ, c[0x0][0x170], PT
ISETP.GT.U32.AND P1, PT, R7, R2, PT
@!P1  IMAD.IADD R2, R2, 0x1, -R7
@!P2  IADD3 R2, -R2, RZ, RZ
@!P0  LOP3.LUT R2, RZ, c[0x0][0x170], RZ, 0x33, !PT
IMAD.WIDE R2, R2, R5, c[0x0][0x160]
LDG.E R3, [R2.64]
IMAD.WIDE R4, R4, R5, c[0x0][0x168]
STG.E [R4.64], R3

后面暂且不翻译了，因为NCCopy指令多的原因已经出来了，就是取模会使用牛顿迭代法，导致指令数量是CCopy的2.69倍。
## 初步结论

那么其实NCCopy的CPI看起来低的奇怪原因也出来了，其实不是因为它访问快，而是因为它指令多，给把CPI平均下来了。 

那么NCCopy的真正耗时的那些指令的平均时延到底是多少？
![](../../../../accessories/Pasted%20image%2020250302052555.png)
我们通过Warp Stall Sampling可以看到，在86.68%的对NCCopy取样中，warp都是卡在STG这条指令上（没卡在LDG是因为Instruction-level parallelism，LDG发了内存请求就继续了，不会卡在赋值R3，只有R3被用了才会等LDG的内存请求返回，也就是跟STG同一行了）

再回过头来看我们开始统计的表：

|                                              | NCCopy | CCopy  | 比例   |
| -------------------------------------------- | ------ | ------ | ---- |
| Time (us)                                    | 405.92 | 270.69 | 1.5  |
| Warp Cycles Per Executed Instruction (cycle) | 52.98  | 86.79  |      |
| Executed Instruction (M inst.)               | 18.35  | 6.82   | 2.69 |
| Achieved Occupancy (%)                       | 81.61  | 74.19  |      |
| CPI x EI / AO                                | 11.91  | 7.97   | 1.49 |

我们可以大胆假设：NCCopy的STG和LDG造成的时延是总耗时的86.68%，然后他们占总指令数量的5.72%。也就是每条STG或LDG指令平均的时延是：52.98 * 18.35 * 0.8668 / (18.35 * 0.0572) = 802 cycle/inst。

同样的，CCopy的STG和LDG造成的时延是总耗时的90.22%，然后他们占总指令数量的15.38%，也就是每条STG或LDG指令平均的时延是：86.79 * 6.82 * 0.9022 / (6.82 * 0.1538) = 509 cycle/inst。

因此我们可以在开始的表格基础上，对指令进行分类后，整理出如下表格：

|                                                    | NCCopy     | CCopy      | 比例       |
| -------------------------------------------------- | ---------- | ---------- | -------- |
| Time (us)                                          | **405.92** | **270.69** | **1.5**  |
| （读写主存）Warp Cycles Per Executed Instruction (cycle) | **802**    | **509**    | **1.58** |
| （读写主存）Executed Instruction (M inst.)               | 1.05       | 1.05       | 1.00     |
| （读写主存）CPI x EI                                     | **842**    | **534**    | **1.58** |
| （其他指令）Warp Cycles Per Executed Instruction (cycle) | 7.49       | 10.03      | 0.75     |
| （其他指令）Executed  Instruction (M inst.)              | 2.44       | 0.67       | 3.64     |
| （其他指令）CPI x EI                                     | 18.28      | 6.72       | 2.72     |
| Achieved Occupancy (%)                             | **81.61**  | **74.19**  | **1.10** |
| CPI x EI / AO                                      | **11.91**  | **7.97**   | **1.49** |

这样就能很直观的看出，NCCopy因为没有coalescing导致的内存访问时延是CCopy的1.58倍，而且这也是它耗时的主要原因，近似于1.5倍的时间比。


#### 访问两次

![](../../../../accessories/Pasted%20image%2020250302151714.png)

原写法（`out[index]=in[index]`）
![](../../../../accessories/Pasted%20image%2020250302151641.png)

访问两次（`out[index]=in[index];out[index]=in[index]+1;`）
![](../../../../accessories/Pasted%20image%2020250302151651.png)

可以看到，第二次访问时`in[index]+1`都碰撞到cache里了。