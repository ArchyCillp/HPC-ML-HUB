Tilelang 找不到 llvm 怎么办? 
`~/tilelang/build$ cmake ..` 出现如下的报错:
![[Pasted image 20250414200515.png]]
两种解决方法: 
第一种: cmake 的时候 specify `-D....`: cmake .. -DTVM_PREBUILD_PATH=/your/path/to/tvm/build
第二种: 直接在 CMakeLists.txt 里面直接 set variable. 
![[Pasted image 20250414204108.png]]
但上面这两种 examples 都是 specify TVM 的 prebuild path. 如果要解决上面 LLVM 的那个报错 error 就需要先install llvm. 

最新的 tvm 貌似把 relay 给改了 (自作聪明下载了最新的 tvm, 实际上得用 tilelang authors 的 fork才行), https://github.com/apache/tvm/pull/17733/files, 从 `src/relay/backend` 改成了 `src/relax/backend`. 导致 tilelang make 的时候会出现下面的错误:
![[Pasted image 20250415105452.png]]
所以这种情况下就不能采用 Install from Source (Using Your Own TVM Installation)
`tilelang/CMakeLists.txt` 里面和 TVM 相关的 variable:
- `TVM_PREBUILD_PATH`: 通常来说是 `tvm/build`
- `TVM_SOURCE_DIR`: 通常来说是 `tvm`
只能用 Install from Source (using the bundled TVM submodule):
但是这个 bundled TVM submodule 在 build 的时候貌似找不到 llvm:
![[Pasted image 20250415164847.png]]
`tilelang/CMakeLists.txt` 里面和 TVM 相关的 commands: 
- Effect of `add_subdirectory(${TVM_SOURCE_DIR} tvm EXCLUDE_FROM_ALL)`:
	1. Includes the TVM source code directory (defined by ${TVM_SOURCE_DIR}) in the build process
	2. Places the build files for TVM in a subdirectory named tvm within the build directory
	3. Uses EXCLUDE_FROM_ALL to prevent TVM targets from being built by default - they'll only be built when explicitly requested or when needed as dependencies

![[Pasted image 20250416175032.png]]
![[Pasted image 20250416175231.png]]

fix NVRTC, CUDA Not Found. 
![[Pasted image 20250416180954.png]]



![[Pasted image 20250416190645.png]]
