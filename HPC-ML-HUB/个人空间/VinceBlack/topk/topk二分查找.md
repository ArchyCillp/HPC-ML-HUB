- 目标：有N个vector，每个vector的dimension是M（<=1024），要求每个vector返回topk的dimension
- 步骤：
	- 每个vector global -> shared memory
	- 每个warp负责一个vector
	- 每个warp reduce 每个vector 的 max min
	- 每个warp二分mid，每个iteration
		- reduce每个vector >= mid 的 count
	- 最后ballot+popcnt扫一遍output >= 的dimension的具体值和index



- 思考，如果只有一个vector，也可以用类似的方法，切分成子vector，一轮下来剩余的vector长度是warp数量 * k，然后最后再搞一遍。