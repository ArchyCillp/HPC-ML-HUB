总体来说链接 cuda 和 python 就是下面这两种方法, 然后我用的是 cmake_example 的方法: 
The [python_example](https://github.com/pybind/python_example) and [cmake_example](https://github.com/pybind/cmake_example) repositories are also a good place to start. They are both complete project examples with cross-platform build systems. The only difference between the two is that [python_example](https://github.com/pybind/python_example) uses Python’s `setuptools` to build the module, while [cmake_example](https://github.com/pybind/cmake_example) uses CMake (which may be preferable for existing C++ projects).

首先得保证 setup.py 和 CMakeLists.txt 在同级目录 (i.e., `cuvs/examples/cpp`)下
然后在 setup.py 那个目录下 `pip install .`, 然后进 cpp/src 里面跑那个 test.py 就行, 入口是在 `src/ivf_flat_16_binder.cu`, module 是在 `src/ivf_flat_16.cu`lib就是在 `src/ivf_flat_16.cuh`

1. Index* Index(int n_clusters)
初始化 index， 输入是聚类数量
使用 `get_index` 接口
2. Index->build_index(uint16_t* keys, uint16_t* values, int seq_len, int n_clusters)
输入是torch gpu tensor底层的data_ptr, 指向 seq_len*128 个 fp16 值 (一个head的keys), 由于c++不支持fp16, 我会cast成uint16_t的指针。输入有keys & values，对keys做KMeans，对values按照KMeans计算的cluster求每个cluster的value sum。
使用 `build_global`, `build_global_multistream`, `build_segment_local`, `build_segment_local_multistream`, `build_segment_global`, `build_segment_global_multistream`. 目前还没 support get value sum. 
3. get_labels()
使用 `index.train_labels()` 即可得到 `raft::device_vector<uint32_t, int64_t>`
4. get_num_valid_cluster()
使用 `index.n_lists()` 即可得到 `uint32_t`
5. get_centroids() 
使用 `index.centers()` in `ivf_flat.hpp` 即可以得到 `raft::device_matrix_view<float, uint32_t, raft::row_major>`
6. 释放 index 资源
使用 `void reset_index(const raft::resources& res, index<half, int64_t>* index);` in `ivf_flat.hpp`, 然后 `delete index;`