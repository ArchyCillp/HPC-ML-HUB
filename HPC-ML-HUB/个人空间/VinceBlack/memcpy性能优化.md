#### nvcomp之大量64KB chunk的memcpy
- nvcomp的接口要求把需要压缩的数据划分成64KB的chunk，无论压缩前还是压缩后都需要以这个单位进行传输，因此就产生了大量的小memcpy
- A10实测，传输一个1.46GB的float数组
	- 如果整个发送，速度为22.67GB/s
	- 如果分chunk，每个chunk单独调用memcpy，整体速度为11.62GB/s
- 还是尽可能整体发送，GPU端生成的数据也应该尽可能在GPU上整理好结构传输回来，毕竟GPU的带宽比PCIe和CPU内存带宽还是高一个数量级的。