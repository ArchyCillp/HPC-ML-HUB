## 解决的问题

- 对于包含 us级别的小kernel但是kernel数量特别多 的任务，CPU端launch kernel的时间可能成了bottleneck
- CUDA Graph可以把 很多kernel的执行顺序 做成所谓的graph直接发给GPU端，避免了大量单kernel的launch时间浪费

## Example
```C++
#define NSTEP 1000
#define NKERNEL 20

// start CPU wallclock timer
for(int istep=0; istep<NSTEP; istep++){
  for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
    shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
    cudaStreamSynchronize(stream);
  }
}
//end CPU wallclock time
```

![[Pasted image 20250227174650.png]]
- 可以看到这个简单的小kernel连续执行（没有memcpy）中间会有较大的CPU CUDA API引起的空隙（虽然时间上很小，但是GPU上执行的更快，CPU端反而成了bottleneck）
- 这个空隙内，CPU端launch了kernel，开启了同步
- kernel在GPU端端执行时间是2.9us，但端到端的kernel平均执行时间却有9.6us


![[Pasted image 20250227175443.png]]
- 把`cudaStreamSynchronize`移出最内层循环后，情况改善了一些，CPU端可以在上一个kernel未结束时就可以launch下一个kernel，从而hide了CPU launch kernel的时间
- 但是kernel之间还是有少许空隙，平均kernel时间3.8us

```C++
bool graphCreated=false;
cudaGraph_t graph;
cudaGraphExec_t instance;
for(int istep=0; istep<NSTEP; istep++){
  if(!graphCreated){
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
      shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
    }
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    graphCreated=true;
  }
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);
}
```
- CUDA Graph可以提前计算好kernel在stream中的执行顺序，一次提交，把launch kernel的任务全都扔给了GPU
- 创建一次Graph的时间可能有点长，但是你可以复用Graph，减少了launch重复多kernel任务的时间浪费