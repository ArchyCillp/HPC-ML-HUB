to deploy a machine learning model. 
TVM should be able to detect the device. 
https://tvm.apache.org/docs/v0.13.0/install/from_source.html


conda create -n tvm-build -c conda-forge \
    "llvmdev>=15" \
    "cmake>=3.24" \
    git \
    python=3.11
## Shared Lib
shared lib for C++ codes:
- `.so` for linux/osx
- `.dylib` for macos
- `dll` for windows

## Install LLVLM
TVM members highly recommend to build with LLVM to enable all the features. 

Install by `apt` might install low version of LLVM. Don't install low version of LLVM. 
LLVM takes long time to build from source, you can download pre-built version of LLVM from https://github.com/llvm/llvm-project/releases, then unzip to a certain location; add `LLVM_CONFIG=/path/to/your/llvm/bin/llvm-config`

We can also install by using `LLVM Nightly Ubuntu Build`, can specify the version number by this method. 

## Building TVM
log shown when run `cmake ..`:
- LLVM libdir: /home/v-xle/miniconda3/envs/tvm-build/lib
- LLVM cmakedir: /home/v-xle/miniconda3/envs/tvm-build/lib/cmake/llvm

`sudo apt-get install zlib1g-dev`: 


`sudo find / -name "libz.so" | head -5`: 
(tvm-build) v-xle@microsoft.com@GCRAZGDL1496:~/tvm/build$ sudo find / -name "libz.so" | head -5
find: ‘/proc/305441’: No such file or directory/home/v-xle/miniconda3/pkgs/zlib-1.2.13-h5eee18b_1/lib/libz.so

**/home/v-xle/miniconda3/lib/libz.so**
/home/v-xle/miniconda3/envs/faiss/lib/libz.so
find: /home/v-xle/miniconda3/envs/kdv/lib/libz.so
‘/proc/305442’/home/v-xle/miniconda3/envs/tilelang/lib/libz.so
: No such file or directory
find: ‘/proc/305448’: No such file or directory

`set(ZLIB_ROOT /home/v-xle/miniconda3/lib/)` can fix the following:
CMake Error at /home/v-xle/miniconda3/envs/tvm-build/share/cmake-4.0/Modules/FindPackageHandleStandardArgs.cmake:227 (message):
  Could NOT find ZLIB (missing: ZLIB_LIBRARY) (found version "1.2.11")

build 完之后需要 export 一下:
- `export TVM_HOME=/path-to-tvm`
- `export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH`

### Correct States From Building TVM
(tvm-build) v-xle@microsoft.com@GCRAZGDL1496:~/tvm/build$ python -c "import tvm; print(tvm._ffi.base._LIB)"
<CDLL '/home/v-xle/tvm/build/libtvm.so', handle 60321d1fc640 at 0x7dc613071b10>
最后的 lib 应该是一个 so 文件. 
在 tvm/build 下面我们可以看到一些 tvm 的 lib:
![[Pasted image 20250414203549.png]]
