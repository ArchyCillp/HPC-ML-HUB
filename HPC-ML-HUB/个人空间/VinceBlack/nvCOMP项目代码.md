---
title: nvCOMP项目代码
---

## high-level API 概念
```C++
LZ4Manager nvcomp_manager{chunk_size, format_opts, stream};
CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);

uint8_t* comp_buffer;
CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));

nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);
```
- LZ4压缩
- 把device memory中的数据和长度，交给LZManager进行压缩准备
	- LZManager需要一个默认64KB的chunk设置，暂且不知道干啥的
- LZManager会返回一个CompressionConfig，里面的max_compressed_buffer_size是压缩数据时需要的最大显存量（压缩后的数据+中间数据），需要给comp_buffer分配这么多显存
- 压缩后的数据在comp_buffer里（也就是comp_buffer是不满的？只有最前面放了压缩后的数据）

```C++
auto decomp_nvcomp_manager = create_manager(comp_buffer, stream);

DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
uint8_t* res_decomp_buffer;
CUDA_CHECK(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));

decomp_nvcomp_manager->decompress(res_decomp_buffer, comp_buffer, decomp_config);
```
- 解压
- 同样用LZManager，交给它包含了压缩数据的comp_buffer（不用长度）
- LZManager会返回一个DecompressionConfig包含了decomp_data_size来构建解压的buffer
- 最终解压的数据在res_decomp_buffer里

## low-level API 概念
```C++
void execute_example(char* input_data, const size_t in_bytes)
{
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	
	size_t* host_uncompressed_bytes;
	const size_t chunk_size = 65536;
	const size_t batch_size = (in_bytes + chunk_size - 1) / chunk_size;
	
	char* device_input_data;
	cudaMalloc(&device_input_data, in_bytes);
	cudaMemcpyAsync(device_input_data, input_data, in_bytes, cudaMemcpyHostToDevice, stream);
	
	cudaMallocHost(&host_uncompressed_bytes, sizeof(size_t)*batch_size);
	for (size_t i = 0; i < batch_size; ++i) {
		if (i + 1 < batch_size) {
		  host_uncompressed_bytes[i] = chunk_size;
		} else {
		  // last chunk may be smaller
		  host_uncompressed_bytes[i] = in_bytes - (chunk_size*i);
		}
	}
	
	// Setup an array of pointers to the start of each chunk
	void ** host_uncompressed_ptrs;
	cudaMallocHost(&host_uncompressed_ptrs, sizeof(size_t)*batch_size);
	for (size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
		host_uncompressed_ptrs[ix_chunk] = device_input_data + chunk_size*ix_chunk;
	}
```
- 把输入数据按照64KB的大小分成chunk，chunk的数量叫做batch_size (?)
- 异步copy全部输入数据到GPU
- `host_uncompressed_bytes[i]` 记录每个batch (chunk)的大小，除了最后一个都是64KB
- `host_uncompressed_ptrs[i]` 记录指向每个batch在device上的起始位置

```C++
	size_t* device_uncompressed_bytes;
	void ** device_uncompressed_ptrs;
	cudaMalloc(&device_uncompressed_bytes, sizeof(size_t) * batch_size);
	cudaMalloc(&device_uncompressed_ptrs, sizeof(size_t) * batch_size);
	
	cudaMemcpyAsync(device_uncompressed_bytes, host_uncompressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(device_uncompressed_ptrs, host_uncompressed_ptrs, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
	
	// Then we need to allocate the temporary workspace and output space needed by the compressor.
	size_t temp_bytes;
	nvcompBatchedLZ4CompressGetTempSize(batch_size, chunk_size, nvcompBatchedLZ4DefaultOpts, &temp_bytes);
	void* device_temp_ptr;
	cudaMalloc(&device_temp_ptr, temp_bytes);
	
	// get the maxmimum output size for each chunk
	size_t max_out_bytes;
	nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunk_size, nvcompBatchedLZ4DefaultOpts, &max_out_bytes);
	
	// Next, allocate output space on the device
  void ** host_compressed_ptrs;
  cudaMallocHost(&host_compressed_ptrs, sizeof(size_t) * batch_size);
  for(size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
      cudaMalloc(&host_compressed_ptrs[ix_chunk], max_out_bytes);
  }

  void** device_compressed_ptrs;
  cudaMalloc(&device_compressed_ptrs, sizeof(size_t) * batch_size);
  cudaMemcpyAsync(
      device_compressed_ptrs, host_compressed_ptrs, 
      sizeof(size_t) * batch_size,cudaMemcpyHostToDevice, stream);

  // allocate space for compressed chunk sizes to be written to
  size_t * device_compressed_bytes;
  cudaMalloc(&device_compressed_bytes, sizeof(size_t) * batch_size);

// And finally, call the API to compress the data
  nvcompStatus_t comp_res = nvcompBatchedLZ4CompressAsync(
      device_uncompressed_ptrs,
      device_uncompressed_bytes,
      chunk_size, // The maximum chunk size
      batch_size,
      device_temp_ptr,
      temp_bytes,
      device_compressed_ptrs,
      device_compressed_bytes,
      nvcompBatchedLZ4DefaultOpts,
      stream);

```
- batch相关的metadata移到device
- `nvcompBatchedLZ4CompressGetTempSize` 根据每个batch的大小（chunk_size）和batch数量，计算出所需要的临时内存
- `nvcompBatchedLZ4CompressGetMaxOutputChunkSize` 计算压缩后的每个batch的最大大小，在device上为压缩后的数据分配内存
- `nvcompBatchedLZ4CompressAsync`最终执行压缩，输入数据是batch_size个最大chunk_size大小的batch，可以使用`device_temp_ptr`的临时内存，压缩后的batch存放于`device_compressed_ptrs`指向的device内存中
