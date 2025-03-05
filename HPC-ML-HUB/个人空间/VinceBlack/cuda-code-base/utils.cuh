#pragma once
#include <cuda_runtime.h>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <iostream>
#include <stdexcept>

#define CHKERR(call)                                          \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)


namespace mycudabook{

/**
 * Example usage:
 * CudaGraphManager manager; 
 * manager.beginCapture();
 * manager.recordEvent("mykernel_start");
 * mykernel<_, _, _, manager.getStream()>(...);
 * manager.recordEvent("mykernel_end");
 * manager.endCaptureAndRun();
 * fmt::print("{} ms",manager.getElapsedTime("mykernel_start","mykernel_end"));
 */
class CudaGraphManager {
    public:
        CudaGraphManager() : stream(nullptr), graph(nullptr), instance(nullptr) {}
    
        ~CudaGraphManager() {
            if (stream) cudaStreamDestroy(stream);
            if (graph) cudaGraphDestroy(graph);
            if (instance) cudaGraphExecDestroy(instance);
            for (auto& event_pair : events) {
                cudaEventDestroy(event_pair.second);
            }
        }
    
        // 开始捕获 CUDA Graph
        void beginCapture() {
            CHKERR(cudaStreamCreate(&stream));
            CHKERR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        }
    
        // 结束捕获并执行 CUDA Graph
        void endCaptureAndRun() {
            CHKERR(cudaStreamEndCapture(stream, &graph));
            CHKERR(cudaGraphInstantiate(&instance, graph));
            CHKERR(cudaGraphLaunch(instance, stream));
            CHKERR(cudaStreamSynchronize(stream));
        }
    
        // 记录事件开始或结束
        void recordEvent(const std::string& eventName) {
            cudaEvent_t event;
            CHKERR(cudaEventCreate(&event));
            CHKERR(cudaEventRecordWithFlags(event, stream, cudaEventRecordExternal));
            events[eventName] = event;
        }
    
        // 计算事件时间差
        float getElapsedTime(const std::string& startEventName, const std::string& endEventName) {
            float milliseconds = 0;
            CHKERR(cudaEventElapsedTime(&milliseconds, events[startEventName], events[endEventName]));
            return milliseconds;
        }
    
        // 获取当前的 CUDA Stream
        cudaStream_t& getStream() const {
            return stream;
        }
    
    private:
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t instance;
        std::unordered_map<std::string, cudaEvent_t> events;
    };
    
    
    class CudaMemoryManager {
    public:
        CudaMemoryManager() {
            // No default stream is created initially
        }
    
        ~CudaMemoryManager() {
            // Clean up all allocated device memory
            // Destroy all streams
            for (cudaStream_t stream : streams_) {
                checkCudaError(cudaStreamDestroy(stream), "Stream destruction failed");
            }
        }
    
        // Allocate device memory and copy data asynchronously in a new stream
        void asyncCopyToDevice(void* & dev_ptr, const void* host_ptr, size_t size) {
            // Create a new stream for this operation
            cudaStream_t stream;
            checkCudaError(cudaStreamCreate(&stream), "Stream creation failed");
            streams_.push_back(stream); // Remember the stream
    
            // Copy data asynchronously in the new stream
            checkCudaError(cudaMemcpyAsync(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync failed");
        }

        // Allocate device memory and copy data asynchronously in a new stream
        void asyncMalloc(void* & dev_ptr, size_t size) {
            // Create a new stream for this operation
            cudaStream_t stream;
            checkCudaError(cudaStreamCreate(&stream), "Stream creation failed");
            streams_.push_back(stream); // Remember the stream
    
            // Copy data asynchronously in the new stream
            checkCudaError(cudaMalloc(dev_ptr, size), "cudaMalloc Failed");}
    
        // Synchronize all streams to ensure all tasks are completed
        void synchronize() {
            for (cudaStream_t stream : streams_) {
                checkCudaError(cudaStreamSynchronize(stream), "Stream synchronization failed");
            }
        }
    
    private:
        std::vector<cudaStream_t> streams_; // Track all streams
    
        // Helper function to check for CUDA errors
        void checkCudaError(cudaError_t err, const char* msg) {
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
            }
        }
    };    
}