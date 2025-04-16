To use CUDA, you will need to bind the CPU based API.  
The workflow will look like this: `python -> pybinding -> c++ -> CUDA`: 
https://forums.developer.nvidia.com/t/how-let-python-invoke-cuda-dynamic-link-library-so-file/154965/4

Tutorial: https://chatgpt.com/share/67f3586d-6758-8011-afdd-669777de2330

The `python3 -m pybind11 --includes` command fetches the include paths for both pybind11 and Python headers.

`nvcc --shared -Xcompiler -fPIC cuda_code.cu -o cuda_code.dll`
逐个参数解释如下：

- **nvcc**  
    CUDA 的编译器驱动程序，用于编译 CUDA 程序。它会将 CUDA 代码（.cu 文件）分离出设备代码和主机代码，并调用合适的编译器进行编译。
    
- **--shared**  
    指示 nvcc 生成共享库（dynamic/shared library）。在 Windows 下，生成的共享库扩展名为 `.dll`；在 Linux 下则通常为 `.so`。
    
- **-Xcompiler -fPIC**  
    这个参数的作用可以拆分为两部分：
    
    - **-Xcompiler**：告诉 nvcc，将紧跟其后的选项直接传递给主机编译器，而不是 nvcc 自己处理。
        
    - **-fPIC**：这是一个主机编译器选项，意思是生成“位置无关代码”（Position Independent Code），这对于共享库来说非常重要，因为它使得生成的代码可以加载到内存中的任意地址。
        
        > 注意：虽然 `-fPIC` 在 Linux 环境下很常用，但在 Windows 下一般不需要，因为 Windows 的编译器有其他机制处理位置无关代码。不过在一些跨平台或者特定配置下，可能会看到这样的参数。
        
- **cuda_code.cu**  
    指定要编译的源文件，即包含 CUDA 代码的文件。
    
- **-o cuda_code.dll**  
    指定编译生成的输出文件名为 `cuda_code.dll`，这就是最终生成的动态链接库文件。



我现在正在尝试用 python 把我的 .cu 文件里面的几个重要函数给 warp 起来. 我看到网上有一种用 pybind11 的方法. 然后网上给了一些 tutorial, 关于告诉如何 Building with CMake的, 我把这个 tutorial 作为 pdf 附件的形式传给你. 然后请你告诉我我该如何修改我的 CMakeLists.txt 去 build 我现有的 ivf_flat_fp16_example.cu 文件然后去 warp 里面的 build_global 函数提供给 python 使用. 我该执行什么样的指令去在 python 里面调用这个 build_global? 我是否需要一些 setup 操作, 比如说 build 完  CMakeLists.txt 之后是否需要一些 setup 然后我的 python 才能 import 这个函数来使用? 据我所知 python 里面传给 cuda 的数据是以 torch::Tensor的形式传的, 比如 input_keys = torch.randn((kv_head_num, seq_len, dim), dtype=torch.float16, device='cuda'), c = my_kernel.kernel_load(input_keys, input_values, n_clusters, n_segments) 这样子.

我的 ivf_flat_fp16_example.cu 内容如下:

ivf_flat_fp16_example.cu 所属的 CMakeLists.txt 内容如下:
 
然后我还在网上看到有类似讨论的例子, 好像比较相关, 但不知道对不对, 我也提供给你:

https://discuss.pytorch.org/t/cannot-build-pybind11-libtorch-code-with-cmake/94377
https://answers.ros.org/question/362178/

Cannot build pybind11/libtorch code with cmake: 
https://discuss.pytorch.org/t/cannot-build-pybind11-libtorch-code-with-cmake/94377



总体来说链接 cuda 和 python 就是下面这两种方法, 然后我用的是 cmake_example 的方法: 
The [python_example](https://github.com/pybind/python_example) and [cmake_example](https://github.com/pybind/cmake_example) repositories are also a good place to start. They are both complete project examples with cross-platform build systems. The only difference between the two is that [python_example](https://github.com/pybind/python_example) uses Python’s `setuptools` to build the module, while [cmake_example](https://github.com/pybind/cmake_example) uses CMake (which may be preferable for existing C++ projects).

bg we have to know: https://packaging.python.org/en/latest/discussions/setup-py-deprecated/

首先得保证 setup.py 和 CMakeLists.txt 在同级目录下. 
用 `python setup.py install` 之后不知道为什么装的是 egg 文件, egg 文件好像已经是 deprecated (如下):
![[Pasted image 20250408135622.png]]
建议是通过 `pip install .` 来执行
安装过程中会正确触发 CMake 构建，编译 CUDA 代码生成 `.so` 文件，并将其置于打包路径下。安装完成后，`.so` 模块已位于 Python 的包目录中，直接 `import ivf_flat_16p` 即可使用，无需手动配置路径。这满足了“安装后直接可用”的要求。
使用 `pip install .` 时，pip 将尝试构建本项目的 wheel 然后安装之。这一过程会自动调用 `setup.py` 的构建逻辑，但最终以 **.whl** 格式安装，从而满足避免 egg、以wheel方式安装的要求.
![[Pasted image 20250408152838.png]]

其次就是得保证
![[Pasted image 20250408120526.png]]
的名字和
![[Pasted image 20250408120453.png]]
名字一样.

跑完这个 CMakeLists.txt 之后会生成类似这样的 .so 文件, 然后这个 .so 文件会被放到 site-packages folder 下面. 
![[Pasted image 20250408121237.png]]

![[Pasted image 20250408114237.png]]


你能比较详细地告诉我这个 setup.py 具体是在做什么吗? 我在和这个 setup.py 的同个目录下执行 `pip install .` 会发生什么? 我这个 setup.py 的同级目录下还有一个 CMakeLists.txt. 这个 CMakeLists.txt 之前是被 CMakeLists.txt 的上一级目录里面的 build.sh 去调用执行的, 这个 build 目前是可以成功执行的. 就是我之前跑 setup.py 的时候 (i.e., through python setup.py install, 会给我安装一堆egg, 和一些 deprecated 的东西), 我想我这次会用 `pip install .`, 应该就会给我安装 whl 文件, 但还会不会安装一些 deprecated 的东西, 我不是很清楚, 我将会给你提供 setup.py, CMakeLists.txt, 和 build.sh, 请你给我分析一下我是否要修改 setup.py 去让 whl 正确地去 install 同时还不会给我安装一些 deprecated 的东西. 
以下是 setup.py 的内容:
一下是 CMakeLists.txt 的内容:
一下是 build.sh 的内容:





尝试去 add cuvs 的 static lib
![[Pasted image 20250408182431.png]]



undefined symbol: _ZN8pybind116detail11type_casterIN2at6TensorEvE4loadENS_6handleEb
indicates that the conversion function for the PyTorch tensor type (as expected by pybind11) is missing at runtime. This is most often caused by mismatches in library versions, inconsistent ABI settings, or linking errors where the module was built against a different version than what is being used. By ensuring that your build environment, dependency versions, and ABI configurations are aligned, you can typically resolve this issue.

![[Pasted image 20250409093814.png]]



ldd /home/v-xle/miniconda3/envs/cuvs/lib/python3.12/site-packages/ivf_flat_16p.cpython-312-x86_64-linux-gnu.so
        linux-vdso.so.1 (0x00007ffc693f7000)
        libtorch.so => /home/v-xle/miniconda3/envs/cuvs/lib/python3.12/site-packages/torch/lib/libtorch.so (0x0000735deef28000)
        libc10.so => /home/v-xle/miniconda3/envs/cuvs/lib/python3.12/site-packages/torch/lib/libc10.so (0x0000735deee26000)
        libpython3.12.so.1.0 => /home/v-xle/miniconda3/envs/cuvs/lib/libpython3.12.so.1.0 (0x0000735dee600000)
        libtorch_cpu.so => /home/v-xle/miniconda3/envs/cuvs/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so (0x0000735dda400000)
        libtorch_cuda.so => /home/v-xle/miniconda3/envs/cuvs/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so (0x0000735da5600000)
        libcudart.so.12 => /home/v-xle/miniconda3/envs/cuvs/lib/libcudart.so.12 (0x0000735da5000000)
        libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x0000735deee14000)
        libstdc++.so.6 => /home/v-xle/miniconda3/envs/cuvs/lib/libstdc++.so.6 (0x0000735da541e000)
        libgcc_s.so.1 => /home/v-xle/miniconda3/envs/cuvs/lib/libgcc_s.so.1 (0x0000735deedf5000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x0000735da4c00000)
        libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x0000735deedee000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x0000735deed05000)
        /lib64/ld-linux-x86-64.so.2 (0x0000735def110000)
        libutil.so.1 => /lib/x86_64-linux-gnu/libutil.so.1 (0x0000735deed00000)
        librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x0000735deecfb000)
        libgomp-24e2ab19.so.1 => /home/v-xle/miniconda3/envs/cuvs/lib/python3.12/site-packages/torch/lib/libgomp-24e2ab19.so.1 (0x0000735da4600000)
        libcupti.so.12 => /home/v-xle/miniconda3/envs/cuvs/lib/python3.12/site-packages/torch/lib/../../nvidia/cuda_cupti/lib/libcupti.so.12 (0x0000735da3e00000)
        libc10_cuda.so => /home/v-xle/miniconda3/envs/cuvs/lib/python3.12/site-packages/torch/lib/libc10_cuda.so (0x0000735dee54f000)
        libcusparse.so.12 => /home/v-xle/miniconda3/envs/cuvs/lib/python3.12/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12 (0x0000735d92000000)
        libcufft.so.11 => /home/v-xle/miniconda3/envs/cuvs/lib/python3.12/site-packages/torch/lib/../../nvidia/cufft/lib/libcufft.so.11 (0x0000735d80c00000)
        libcusparseLt.so.0 => /home/v-xle/miniconda3/envs/cuvs/lib/python3.12/site-packages/torch/lib/../../cusparselt/lib/libcusparseLt.so.0 (0x0000735d72200000)
        libcurand.so.10 => /home/v-xle/miniconda3/envs/cuvs/lib/python3.12/site-packages/torch/lib/../../nvidia/curand/lib/libcurand.so.10 (0x0000735d6ba00000)
        libcublas.so.12 => /home/v-xle/miniconda3/envs/cuvs/lib/python3.12/site-packages/torch/lib/../../nvidia/cublas/lib/libcublas.so.12 (0x0000735d64e00000)
        libcublasLt.so.12 => /home/v-xle/miniconda3/envs/cuvs/lib/python3.12/site-packages/torch/lib/../../nvidia/cublas/lib/libcublasLt.so.12 (0x0000735d43200000)
        libcudnn.so.9 => /home/v-xle/miniconda3/envs/cuvs/lib/python3.12/site-packages/torch/lib/../../nvidia/cudnn/lib/libcudnn.so.9 (0x0000735d42c00000)
        libnccl.so.2 => /home/v-xle/miniconda3/envs/cuvs/lib/python3.12/site-packages/torch/lib/../../nvidia/nccl/lib/libnccl.so.2 (0x0000735d34200000)
        libnvJitLink.so.12 => /home/v-xle/miniconda3/envs/cuvs/lib/python3.12/site-packages/torch/lib/../../nvidia/cusparse/lib/../../nvjitlink/lib/libnvJitLink.so.12 (0x0000735d30c00000)
        



TODO:
大概正确的话数据就是这个值. 
![[Pasted image 20250409154212.png]]
