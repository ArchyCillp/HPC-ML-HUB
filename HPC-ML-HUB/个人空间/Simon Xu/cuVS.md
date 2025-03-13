## Kmeans
first look at https://github.com/rapidsai/cuvs/tree/branch-25.04/cpp/src/cluster
The main files to look at for implementing fp16 support in balanced kmeans are:

1. First, you need to create a new implementation file for the fp16 version, similar to the existing files:

- kmeans_balanced_fit_float.cu → kmeans_balanced_fit_half.cu

- kmeans_balanced_fit_predict_float.cu → kmeans_balanced_fit_predict_half.cu

- kmeans_balanced_predict_float.cu → kmeans_balanced_predict_half.cu

1. The implementation already has support for half precision in the data type mapping (config<half> is already defined in ann_utils.cuh), but the actual implementation files for half precision are missing.

To implement fp16 support, you would need to:

1. Create the new implementation files mentioned above

2. Use the cuvs::spatial::knn::detail::utils::mapping<float>{} operator for conversion from half to float (since the core algorithm works with float)

For example, a new kmeans_balanced_fit_half.cu file would look like:

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