---
title: map & set
---

## 如何对map和set检查存在、添加、删除元素？
```C++
// Map example
std::map<int, std::string> myMap;
myMap[1] = "Apple";
myMap[2] = "Banana";

if (myMap.find(1) != myMap.end()) std::cout << "Key 1 exists in map." << std::endl;
myMap[3] = "Cherry";
myMap.erase(2);

// Set example
std::set<int> mySet;
mySet.insert(10);
mySet.insert(20);

if (mySet.find(10) != mySet.end()) std::cout << "Value 10 exists in set." << std::endl;
mySet.insert(30);
mySet.erase(20);

```


## 如何遍历一个map的key，value？
```C++
// Iterate keys and values
for (const auto& [key, value] : myMap) {
    // Use key and value
}

// Iterate keys
for (const auto& [key, _] : myMap) {
    // Use key
}

// Iterate values
for (const auto& [_, value] : myMap) {
    // Use value
}
```


## 如何嵌套遍历一个map？
```C++

for (auto it = myMap.begin(); it != myMap.end(); ++it) {
	std::cout << "Current key: " << it->first << ", value: " << it->second << std::endl;

	// Nested loop from the current key to the end
	for (auto nested_it = it; nested_it != myMap.end(); ++nested_it) {
		std::cout << "  Nested key: " << nested_it->first << ", value: " << nested_it->second << std::endl;
	}
}
```

## ordered_map(map)和unordered_map的区别？


