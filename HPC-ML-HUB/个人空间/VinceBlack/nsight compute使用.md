https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s22141-what-the-profiler-is-telling-you-how-to-get-the-most-performance-out-of-your-hardware.pdf
![](../../../accessories/Pasted%20image%2020250320193335.png)
## 如何使用 Nsight GUI
### Start Activity
![[8968d0ce89e09a6daa58b95f2b3f8ff8.jpg]]
%C/nvidia/nsight-compute/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export %C/nvidia/nsight-compute/ivf_flat_build_cluster_segment_assignment_local --force-overwrite --section-folder %C/nvidia/nsight-compute/sections --set full /home/v-xle/cuvs/examples/cpp/build/IVF_FLAT_FP16_EXAMPLE

通常来说在 profile 之前还会有一系列的 deployment:
![[b8a7fbc2540d0ea99ff309a375502ac2.jpg]]
