参考 https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master?tab=readme-ov-file
### naive
- 每个thread一个算一个`A[i] * B[i]`，然后atomic加到global c里
- 性能是422 us

### pack half2
- vectorization：每个thread一次请求一个half2
- scalarC用pack的half2
- 性能是137.02 us

### fastAtomicAdd
 - scalarC用pack的half2，也就是x是原scalarC，y一直是0
 - 这样的目的就是避免使用16-bit的atomicAdd而是使用32-bit的atomicAdd
 - 我感觉这应该是16-bit的atomic操作支持较差的原因
 - 性能是137.01 us


