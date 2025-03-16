---
title: HQ's project problem
---

### Key-value atomic increment using hash table:
![](../../../accessories/Pasted%20image%2020250303115613.png)

#### My idea:
signature(keys) = some_hash_function_different_from_index(keys) in range \[1, max_int - 1\]

old = CAS....
if old == 0:
	write keys and set state to signature(keys)
if old == signature(keys):
	check whether the keys == bucket.key
	atomicAdd(bucket.count, 1)
	return true 



