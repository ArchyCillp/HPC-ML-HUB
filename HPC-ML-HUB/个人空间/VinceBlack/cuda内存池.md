---
title: cuda内存池
---

```C++
  

class MY_CUDA_MEMORY_POOL_V1 {

public:

MY_CUDA_MEMORY_POOL_V1(size_t deviceMemorySize=(size_t)4*1024*1024*1024, size_t hostMemorySize=(size_t)8*1024*1024*1024)

: deviceMemorySize_(deviceMemorySize), hostMemorySize_(hostMemorySize), commited_(false) {

fmt::print("Launch CudaMemoryPool with {} MB on device and {} MB on host.\n", deviceMemorySize*1.0/1024/1024, hostMemorySize*1.0/1024/1024);

// Allocate device memory

CHKERR(cudaMalloc(&deviceMemory_, deviceMemorySize_));

// Allocate host memory

CHKERR(cudaMallocHost(&hostMemory_, hostMemorySize_));

}

~MY_CUDA_MEMORY_POOL_V1() {

// Free device memory

cudaFree(deviceMemory_);

// Free host memory

cudaFreeHost(hostMemory_);

}

void wantHost(void** ptr_addr, size_t size) {

if (!commited_){

hostRequests_.push_back({ptr_addr, size});

satisfy();

}

else {

throw std::runtime_error("Cannot wantHost on a commited memory pool.");

}

}

void wantDevice(void** ptr_addr, size_t size) {

if (!commited_){

deviceRequests_.push_back({ptr_addr, size});

satisfy();

}

else {

throw std::runtime_error("Cannot wantDevice on a commited memory pool.");

}

}

bool satisfy() {

if (!commited_) {

cudaDeviceSynchronize();

size_t totalHostRequested = hostOffset_;

for (const auto& req : hostRequests_) {

totalHostRequested += req.size;

}

if (totalHostRequested > hostMemorySize_) {

std::cerr << "Error: Not enough host memory available, desired "<< totalHostRequested

<<" bytes, allocated "<< hostMemorySize_ <<" bytes." << std::endl;

throw std::runtime_error("RUNTIME ERROR");

return false;

}

// Check if the total requested device memory is valid

size_t totalDeviceRequested = deviceOffset_;

for (const auto& req : deviceRequests_) {

totalDeviceRequested += req.size;

}

if (totalDeviceRequested > deviceMemorySize_) {

std::cerr << "Error: Not enough device memory available, desired "<< totalDeviceRequested

<<" bytes, allocated "<< deviceMemorySize_ <<" bytes." << std::endl;

throw std::runtime_error("RUNTIME ERROR");

return false;

}

// 永远牢记：内存是256 aligned!!!!!!!!!!

// Assign host memory addresses

for (auto& req : hostRequests_) {

// Round up hostOffset_ to the nearest multiple of 256

hostOffset_ = (hostOffset_ + 255) & ~255;

*(req.ptr) = static_cast<char*>(hostMemory_) + hostOffset_;

hostOffset_ += req.size;

}

  

// Assign device memory addresses

for (auto& req : deviceRequests_) {

// Round up deviceOffset_ to the nearest multiple of 256

deviceOffset_ = (deviceOffset_ + 255) & ~255;

*(req.ptr) = static_cast<char*>(deviceMemory_) + deviceOffset_;

deviceOffset_ += req.size;

}

hostRequests_.clear();

deviceRequests_.clear();

return true;

}

else {

throw std::runtime_error("Cannot satisfy on a commited memory pool.");

return false;

}

}

bool commit() {

// Check if the total requested host memory is valid

bool res = satisfy();

commited_ = true;

fmt::print("Memmory pool: {} MB host allocated, {} MB device allocated", hostOffset_*1.0/1024/1024, deviceOffset_*1.0/1024/1024);

return res;

}

void releaseAll(){

hostRequests_.clear();

deviceRequests_.clear();

hostOffset_ = 0;

deviceOffset_ = 0;

commited_ = false;

}

private:

struct MemoryRequest {

void** ptr;

size_t size;

};

void* deviceMemory_;

void* hostMemory_;

size_t deviceMemorySize_;

size_t hostMemorySize_;

bool commited_;

size_t hostOffset_ = 0;

size_t deviceOffset_ = 0;

std::vector<MemoryRequest> hostRequests_;

std::vector<MemoryRequest> deviceRequests_;

};
```

几个需要注意的点：
- CUDA内存是256 byte aligned的，也就是任何malloc都是返回256 byte的乘数地址，不符合这个规定很可能出错。
- `cudaMallocHost`非常关键，它能让host上的内存分配的是pinned memory，这种memory不会进swap区，也就是肯定不在硬盘里。因此DMA可以直接启动host<->device的传输。通常的内存是pagable的，也就是进行pcie复制前，需要一个一个page地load到page-locked staging buffer，这就慢了。在GPU01机器上检测到，一个一个多GB的大数组的传输，使用pinned memory可以到24GB/s，而使用pagable memory只有10GB/s。
