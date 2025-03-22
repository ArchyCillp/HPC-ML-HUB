## API Basics
Memory management: 
- cuVS uses the [RMM]([rapidsai/rmm](https://github.com/rapidsai/rmm)) library. 
Here’s an example of configuring RMM to use a pool allocator in C++ (derived from the RMM example [here](https://github.com/rapidsai/rmm?tab=readme-ov-file#example)):
```cpp
rmm::mr::cuda_memory_resource cuda_mr;
// Construct a resource that uses a coalescing best-fit pool allocator
// With the pool initially half of available device memory
auto initial_size = rmm::percent_of_free_device_memory(50);
rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr, initial_size};
rmm::mr::set_current_device_resource(&pool_mr); // Updates the current device resource pointer to `pool_mr`
rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(); // Points to `pool_mr`
```
`rmm::mr::set_current_device_resource(&pool_mr);` 这行代码告诉我们 pool_mr 其实就是 current device resource. 

---
Resource management: 
- cuVS uses an API from the [RAFT](https://github.com/rapidsai/raft) to centralize and reuse expensive resources, such as memory management.
```cpp
#include <raft/core/device_resources.hpp>
raft::device_resources res;
```
### Resources
All resources which are specific to a computing environment like host or device are contained within, and managed by, `raft::resources`.
https://docs.rapids.ai/api/raft/nightly/cpp_api/core_resources/
### Device Resources
`raft::device_resources` is a **convenience over** using `raft::resources` directly. It provides accessor methods to retrieve resources such as the CUDA stream, stream pool, and handles to the various CUDA math libraries like cuBLAS and cuSOLVER.
## Multi-dimensional span (C++)
https://docs.rapids.ai/api/cuvs/nightly/api_interoperability/#multi-dimensional-span-c
https://docs.rapids.ai/api/raft/stable/cpp_api/mdspan/
`mdarray` structure: simplifies the allocation and management of multi-dimensional data in both host and device (GPU) memory.
The `mdarray` is an owning object that forms a convenience layer over RMM and can be constructed in RAFT using a number of different helper functions:
```cpp
#include <raft/core/device_mdarray.hpp>

int n_rows = 10;
int n_cols = 10;

auto scalar = raft::make_device_scalar<float>(handle, 1.0);
auto vector = raft::make_device_vector<float>(handle, n_cols);
auto matrix = raft::make_device_matrix<float>(handle, n_rows, n_cols);
```
The `mdspan` is a lightweight non-owning view that can wrap around any pointer, maintaining shape, layout, and indexing information for accessing elements.
关于`mdspan`, 直接参考 C++ 就行: 
- https://en.cppreference.com/w/cpp/container/mdspan
- std::mdspan for C++23入门 - 南山烟雨珠江潮的文章 - 知乎 https://zhuanlan.zhihu.com/p/653155513

```cpp
// Scalar mdspan on device
auto scalar_view = scalar.view();

// Vector mdspan on device
auto vector_view = vector.view();

// Matrix mdspan on device
auto matrix_view = matrix.view();
```

多维数组`T arr[M][N][O]`的内存是连续的，布局上其实是一个一维数组。mdspan正是基于此原理定义，传入一维数组的首地址，然后再传入每一维的大小，既可以当成一个多维数组使用：
```cpp
```cpp
vector<int> v (M * N * O); // 一维数组
mdspan sp (v.data(), M, N, O); // 当做三维数组使用
// 像这样访问每个元素：
// sp[2, 3, 4] = 5;
```


```cpp
Of course, RAFT’s `mdspan`/`mdarray` APIs aren’t just limited to the `device`. You can also create `host` variants:
```cpp
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>

int n_rows = 10;
int n_cols = 10;

auto scalar = raft::make_host_scalar<float>(handle, 1.0);
auto vector = raft::make_host_vector<float>(handle, n_cols);
auto matrix = raft::make_host_matrix<float>(handle, n_rows, n_cols);

auto scalar_view = raft::make_host_scalar_view(scalar.data_handle());
auto vector_view = raft::make_host_vector_view(vector.data_handle(), vector.extent(0));
auto matrix_view = raft::make_host_matrix_view(matrix.data_handle(), matrix.extent(0), matrix.extent(1));
```
### view()
![[../../../../accessories/Pasted image 20250317193050.png]]
![[../../../../accessories/Pasted image 20250317193040.png]]
### extent()
![[../../../../accessories/Pasted image 20250317192857.png]]
By convention for a row-major layout, “extent(0)” returns the number of rows, and “extent(1)” returns the number of columns.
![[../../../../accessories/Pasted image 20250316234248.png]]
### data()
![[../../../../accessories/Pasted image 20250317192829.png]]

### data_handle()
data_handle() in `mdspan`:  https://en.cppreference.com/w/cpp/container/mdspan
![[../../../../accessories/Pasted image 20250317192258.png]]


只要搞清楚什么是 owning, 什么是 non-owning 就分得清 data(), data_handle() 和 view() 的区别了.


https://chatgpt.com/share/67d70006-fb44-8011-9638-ee49a87641b2


The following are the error log:
(cuvs) simon@simonxcomp:~/test-kmeans/cuvs$ ./build.sh libcuvs
Building for the architecture of the GPU in the system...
-- The CXX compiler identification is GNU 13.3.0
-- The CUDA compiler identification is NVIDIA 12.6.85 with host compiler GNU 13.3.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /home/simon/miniconda3/envs/cuvs/bin/x86_64-conda-linux-gnu-c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /home/simon/miniconda3/envs/cuvs/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Using auto detection of gpu-archs: 86
-- Project CUVS is building for CUDA architectures:
  86-real
-- Found CUDAToolkit: /home/simon/miniconda3/envs/cuvs/targets/x86_64-linux/include (found version "12.6.85")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Check if compiler accepts -pthread
-- Check if compiler accepts -pthread - yes
-- Found Threads: TRUE
-- Found OpenMP_CXX: -fopenmp (found version "4.5")
-- Found OpenMP_CUDA: -fopenmp (found version "4.5")
-- Found OpenMP: TRUE (found version "4.5")
-- Found Git: /usr/bin/git (found version "2.34.1")
-- CPM: Adding package CCCL@2.7.0 (v2.7.0)
CMake Warning (dev) at /home/simon/miniconda3/envs/cuvs/share/cmake-3.31/Modules/FetchContent.cmake:1953 (message):
  Calling FetchContent_Populate(CCCL) is deprecated, call
  FetchContent_MakeAvailable(CCCL) instead.  Policy CMP0169 can be set to OLD
  to allow FetchContent_Populate(CCCL) to be called directly for now, but the
  ability to call it with declared details will be removed completely in a
  future version.
Call Stack (most recent call first):
  build/cmake/CPM_0.40.0.cmake:1074 (FetchContent_Populate)
  build/cmake/CPM_0.40.0.cmake:868 (cpm_fetch_package)
  build/_deps/rapids-cmake-src/rapids-cmake/cpm/find.cmake:187 (CPMAddPackage)
  build/_deps/rapids-cmake-src/rapids-cmake/cpm/cccl.cmake:81 (rapids_cpm_find)
  CMakeLists.txt:178 (rapids_cpm_cccl)
This warning is for project developers.  Use -Wno-dev to suppress it.

[1/9] Creating directories for 'cccl-populate'
[1/9] Performing download step (git clone) for 'cccl-populate'
Cloning into 'cccl-src'...
error: RPC failed; curl 92 HTTP/2 stream 0 was not closed cleanly: CANCEL (err 8)
error: 2219 bytes of body are still expected
fetch-pack: unexpected disconnect while reading sideband packet
fatal: early EOF
fatal: fetch-pack: invalid index-pack output
Cloning into 'cccl-src'...
error: RPC failed; curl 92 HTTP/2 stream 0 was not closed cleanly: CANCEL (err 8)
error: 2068 bytes of body are still expected
fetch-pack: unexpected disconnect while reading sideband packet
fatal: early EOF
fatal: fetch-pack: invalid index-pack output
Cloning into 'cccl-src'...
error: RPC failed; curl 92 HTTP/2 stream 0 was not closed cleanly: CANCEL (err 8)
error: 2169 bytes of body are still expected
fetch-pack: unexpected disconnect while reading sideband packet
fatal: early EOF
fatal: fetch-pack: invalid index-pack output
Had to git clone more than once: 3 times.
CMake Error at cccl-subbuild/cccl-populate-prefix/tmp/cccl-populate-gitclone.cmake:50 (message):
  Failed to clone repository: 'https://github.com/NVIDIA/cccl.git'


FAILED: cccl-populate-prefix/src/cccl-populate-stamp/cccl-populate-download /home/simon/test-kmeans/cuvs/cpp/build/_deps/cccl-subbuild/cccl-populate-prefix/src/cccl-populate-stamp/cccl-populate-download 
cd /home/simon/test-kmeans/cuvs/cpp/build/_deps && /home/simon/miniconda3/envs/cuvs/bin/cmake -DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE -P /home/simon/test-kmeans/cuvs/cpp/build/_deps/cccl-subbuild/cccl-populate-prefix/tmp/cccl-populate-gitclone.cmake && /home/simon/miniconda3/envs/cuvs/bin/cmake -E touch /home/simon/test-kmeans/cuvs/cpp/build/_deps/cccl-subbuild/cccl-populate-prefix/src/cccl-populate-stamp/cccl-populate-download
ninja: build stopped: subcommand failed.

CMake Error at /home/simon/miniconda3/envs/cuvs/share/cmake-3.31/Modules/FetchContent.cmake:1918 (message):
  Build step for cccl failed: 1
Call Stack (most recent call first):
  /home/simon/miniconda3/envs/cuvs/share/cmake-3.31/Modules/FetchContent.cmake:1609 (__FetchContent_populateSubbuild)
  /home/simon/miniconda3/envs/cuvs/share/cmake-3.31/Modules/FetchContent.cmake:2145:EVAL:2 (__FetchContent_doPopulation)
  /home/simon/miniconda3/envs/cuvs/share/cmake-3.31/Modules/FetchContent.cmake:2145 (cmake_language)
  /home/simon/miniconda3/envs/cuvs/share/cmake-3.31/Modules/FetchContent.cmake:1978:EVAL:1 (__FetchContent_Populate)
  /home/simon/miniconda3/envs/cuvs/share/cmake-3.31/Modules/FetchContent.cmake:1978 (cmake_language)
  build/cmake/CPM_0.40.0.cmake:1074 (FetchContent_Populate)
  build/cmake/CPM_0.40.0.cmake:868 (cpm_fetch_package)
  build/_deps/rapids-cmake-src/rapids-cmake/cpm/find.cmake:187 (CPMAddPackage)
  build/_deps/rapids-cmake-src/rapids-cmake/cpm/cccl.cmake:81 (rapids_cpm_find)
  CMakeLists.txt:178 (rapids_cpm_cccl)


-- Configuring incomplete, errors occurred!

![[../../../../accessories/Pasted image 20250319193503.png]]

## Kmeans
first look at https://github.com/rapidsai/cuvs/tree/branch-25.04/cpp/src/cluster

![[../../../../accessories/Pasted image 20250315200750.png]]

![[../../../../accessories/Pasted image 20250315195547.png]]
## Types
### DataT
The input data type. 
### MathT
The centroid type.
### balanced_params
![[../../../../accessories/Pasted image 20250316172608.png]]
### MappingOpT
![[../../../../accessories/Pasted image 20250317155401.png]]
two things need to know:
- `MappingOpT = raft::identity_op`
- `cuvs::spatial::knn::detail::utils::mapping<float>{}` 按理来说这个 mapping 应该是 `raft::identity_op` 的类型
![[../../../../accessories/Pasted image 20250317162425.png]]

需要去提供 raft 和 cuvs 的 link 给 gpt.
I am currently reading the codebase (mixed with cpp and cuda) of cuvs (cuVS is a new library mostly derived from the approximate nearest neighbors and clustering algorithms in the RAPIDS RAFT library of machine learning and data mining primitives.) (https://github.com/rapidsai/cuvs) in github. I am confused about what exactly mapping_op does and what it is used for and how it is used by users.
Could you read the following code and the code in the file that I provide to you to answer my question?



## Balanced KMeans API
### fit
Trains the model to find centroids. 
![[../../../../accessories/Pasted image 20250315201406.png]]
three-stages hierarchical k-means algorithm:
- “Mesocluster” Stage: partitioned into mesoclusters, # mesoclusters is determined as the square root of the final number of clusters. 
- Fine Clustering per Mesocluster: within each mesocluster, further subdivides the points into "fine" clusters. The more data points in a mesocluster, the more fine clusters it has.
- Globally refine centroids with iterations. In each iteration (f):
	- If some clusters have very few points (i.e., are too small), their centroids are shifted toward nearby larger clusters. 
	- 
### predict
Use when you already have trained centroids and want to determine which cluster each data point belongs to. 
![[../../../../accessories/Pasted image 20250315203457.png]]
### fit_predict
Examples of calling `fit_predict`:
```cpp
#include <cuvs/cluster/kmeans_balanced.cuh>
#include <cuvs/cluster/kmeans_balanced_types.hpp>

// ... set up raft resources and data ...
raft::resources handle;
cuvs::cluster::balanced_params params;  // configure hyper-parameters (e.g., n_clusters)
auto centroids = raft::make_device_matrix<float, int>(handle, n_clusters, n_features);
auto labels    = raft::make_device_vector<uint32_t, int>(handle, n_samples);

// X is a raft::device_matrix_view<const float, int> for the input data
cuvs::cluster::kmeans_balanced::fit_predict(handle, params, X, centroids.view(), labels.view());
```


both train the model and get the cluster assignments in one step.
![[../../../../accessories/Pasted image 20250316121545.png]]


Library provides a pre-compiled instance for float data: a specialized `cuvs::cluster::kmeans::fit_predict` for `float`/`uint32_t` that simply calls this balanced version with an appropriate mapping functor​. 
![[../../../../accessories/Pasted image 20250315203844.png]]
how to call `fit_predict`: 
![[../../../../accessories/Pasted image 20250315210031.png]]
will compute `n_clusters` centroids and assign each of the `n_samples` data points a cluster label.


![[../../../../accessories/Pasted image 20250316103941.png]]
![[../../../../accessories/Pasted image 20250316104002.png]]

![[../../../../accessories/Pasted image 20250316104140.png]]




Could you read kmeans_balanced_fit_predict_float.cu, kmeans_balanced.cuh, kmeans.cuh and introduce from a user perspective, how to call fit_predict in kmeans_balanced.cuh for fp32? In the process of calling fit_predict, how many functions/API it will go through, what is the function call flow? In the call flow, please specify the file of each called function. In the call flow, please indicate every API, including the wrapper. 


in the folder of cluster and introduce from a user perspective, how to call fit_predict in kmeans_balanced.cuh for fp32? In the process of calling fit_predict, how many functions it will go through?

## Standard KMeans vs. Balanced KMeans Cluster Sizes

### Standard KMeans

- Points are assigned to their nearest centroid based purely on distance

- No control over cluster sizes

- Can result in highly imbalanced clusters (some very large, some very small)

- The size distribution is completely determined by the data distribution and initial centroid placement

### Balanced KMeans

- Enforces constraints to create more evenly-sized clusters

- Aims for roughly equal number of points in each cluster

- May sacrifice some compactness (distance optimization) to achieve balance

- In an ideal case with n points and k clusters, each cluster would have approximately n/k points

## Prompt 1
I am a beginner to kmenas, could you tell me which file should I look at in this folder if I want to adopt the balanced kmeans to my project. I want to use fp16, but it seems that cuVS only has version of fp32, so I need to figure out how to change it (if possible) and apply it to my project. 

The main files to look at for implementing fp16 support in balanced kmeans are:
```cpp
1. First, you need to create a new implementation file for the fp16 version, similar to the existing files:

- kmeans_balanced_fit_float.cu → kmeans_balanced_fit_half.cu

- kmeans_balanced_fit_predict_float.cu → kmeans_balanced_fit_predict_half.cu

- kmeans_balanced_predict_float.cu → kmeans_balanced_predict_half.cu

1. The implementation already has support for half precision in the data type mapping (config<half> is already defined in ann_utils.cuh), but the actual implementation files for half precision are missing.

To implement fp16 support, you would need to:

1. Create the new implementation files mentioned above

2. Use the cuvs::spatial::knn::detail::utils::mapping<float>{} operator for conversion from half to float (since the core algorithm works with float)

For example, a new kmeans_balanced_fit_half.cu file would look like:

/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// clang-format off
#include "kmeans_balanced.cuh"
#include "../neighbors/detail/ann_utils.cuh"
#include <raft/core/resources.hpp>
// clang-format on

namespace cuvs::cluster::kmeans {

void fit(const raft::resources& handle,
         cuvs::cluster::kmeans::balanced_params const& params,
         raft::device_matrix_view<const half, int> X,
         raft::device_matrix_view<float, int> centroids)
{
  cuvs::cluster::kmeans_balanced::fit(
    handle, params, X, centroids, cuvs::spatial::knn::detail::utils::mapping<float>{});
}
}  // namespace cuvs::cluster::kmeans

Similar implementations would be needed for the predict and fit_predict operations.

The mapping operator is already defined to handle half precision data types, and the scaling factor for half is set to 1.0, just like float and double.

For your project, you have two options:

1. Implement these new files in cuVS and contribute them back to the project

2. In your own codebase, create similar wrapper functions that utilize the existing templated functions from kmeans_balanced.cuh but with half precision data types

If you decide to go with option 2, you'll need to:

1. Include the necessary headers from cuVS

2. Create wrapper functions similar to those in the implementation files but for half precision

3. Call the templated functions with the appropriate mapping operation

Is there a specific aspect of the implementation that you'd like me to explain in more detail?
```
## Prompt 2
Read the following code:

I want to know: 
- in high-level, the major steps of code. 
- purposes/functions of each line of code. 
- motivations of writing code with the given syntax (I am not that familiar with C++/CUDA syntax).


我想知道什么? 

算法实现, especially how to make it parallel. 
每一步的 rationale. 

买拓展坞, 拓展坞有USBC接口, 下载一个软件 (Mos) 把反向效果抵消.