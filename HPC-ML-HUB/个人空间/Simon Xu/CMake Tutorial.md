cmake的执行顺序是从上往下. 
can check cmake variables at cmake doc by https://cmake.org/cmake/help/v3.0/module/FindZLIB.html
对于一个CPP的大项目来说, CMakeLists.txt 都是一层一层的, 外面那层调用里面那层, 要改一些 environmental setting 的话就在最外面那层改就行. 

CMAKE_PREFIX_PATH

CMake Error at 3rdparty/tvm/cmake/utils/FindLLVM.cmake:47 (find_package):
  Could not find a package configuration file provided by "LLVM" with any of
  the following names:
  
    LLVMConfig.cmake
    llvm-config.cmake

  Add the installation prefix of "LLVM" to CMAKE_PREFIX_PATH or set
  "LLVM_DIR" to a directory containing one of the above files.  If "LLVM"
  provides a separate development package or SDK, be sure it has been
  installed.
Call Stack (most recent call first):
  3rdparty/tvm/cmake/modules/LLVM.cmake:31 (find_llvm)
  3rdparty/tvm/CMakeLists.txt:596 (include)

## Useful Commands
`cmake --build . --parallel $(nproc)`
## cmake-commands
### add_subdirectory
### find_library
![[Pasted image 20250416180352.png]]
![[Pasted image 20250416180431.png]]
![[Pasted image 20250416180513.png]]
result:
![[Pasted image 20250416180533.png]]
### if
#### set source dir
classic ways to set SOURCE_DIR:
![[Pasted image 20250416182739.png]]
![[Pasted image 20250416182826.png]]
#### use prebuilt or build from source?
![[Pasted image 20250416185333.png]]

## cmake-properties
### - [EXCLUDE_FROM_ALL](https://cmake.org/cmake/help/latest/prop_tgt/EXCLUDE_FROM_ALL.html)
