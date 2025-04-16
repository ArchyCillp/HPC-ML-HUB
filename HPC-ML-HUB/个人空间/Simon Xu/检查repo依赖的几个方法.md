查看 `.gitmodules`: 
以下是 tilelang 的依赖: 
![[Pasted image 20250415174817.png]]
![[Pasted image 20250415174903.png]]
`git clone --recursive` 基本上就是把 `.gitmodules` 里面的作者维护的原 repo 的 fork 拉了下来.
![[Pasted image 20250415200055.png]]
##  Git submodule
A Git submodule is **a record within a host Git repository that points to a specific commit in another external repository**. Submodules are very static and only track specific commits.
https://github.com/tile-ai/tilelang/issues/13

用 tilelang 的好像 locally 都 maintain 了一个 llvm. 