https://github.com/pytorch/pytorch/issues/108041

首先 dev_resources 一定是要作为参数传进来的
返回的是 gpu 指针.
![[Pasted image 20250405095838.png]]

![[Pasted image 20250404134913.png]]

![[Pasted image 20250404131135.png]]
好像开在 heap 上面是没问题的? 

qianxi 改了这两个地方去多加这个 train_labels_ 的 attribute:
![[Pasted image 20250404132519.png]]

提供一个销毁 index 的接口.


因为 index 是在 build 里面初始化的, 所以我建议, n_clusters 应该作为 build 的一个参数传进去. 而不是创建一个 index constructor. 
![[Pasted image 20250404081638.png]]
其实传一个 index constructor 进来也行. 只要调用 下面这种 build 就行了
![[Pasted image 20250404081817.png]]




3. tuple(int*, int) Index->get_labels()

输入是 index 指针，输出长度为 seq_len 的 int 数组，表示每一个data id到cluster id的映射，cluster id是从0开始，另外一个int返回值表示有效的cluster数目。

3. tuple(uint16_t*, uint16_t*) Index->get_centroids()

输入是 index 指针，输出两个指向长度为 实际cluster数量*128 个 fp16 元素的指针，同样由于c++不支持float16，需要cast成uint16的指针。两个指针分别表示centroids和value sum。


销毁的接口好像找到了 (i.e., release())
![[Pasted image 20250404171652.png]]



![[Pasted image 20250405084539.png]]




1. Index* Index(int n_clusters)
初始化 index， 输入是聚类数量
使用 `get_index` 接口
2. Index->build_index(uint16_t* keys, uint16_t* values, int seq_len, int n_clusters)
输入是torch gpu tensor底层的data_ptr, 指向 seq_len*128 个 fp16 值 (一个head的keys), 由于c++不支持fp16, 我会cast成uint16_t的指针。输入有keys & values，对keys做KMeans，对values按照KMeans计算的cluster求每个cluster的value sum。
使用 `build_global`, `build_global_multistream`, `build_segment_local`, `build_segment_local_multistream`, `build_segment_global`, `build_segment_global_multistream`. 目前还没 support value sum. 
3. get_labels()
使用 `index.train_labels()` 即可得到 `raft::device_vector<uint32_t, int64_t>`
4. get_num_valid_cluster()
使用 `index.n_lists()` 即可得到 `uint32_t`
5. get_centroids() 
使用 `index.centers()` in `ivf_flat.hpp` 即可以得到 `raft::device_matrix_view<float, uint32_t, raft::row_major>`
6. 释放 index 资源
使用 `void reset_index(const raft::resources& res, index<half, int64_t>* index);` in `ivf_flat.hpp`, 然后 `delete index;`



