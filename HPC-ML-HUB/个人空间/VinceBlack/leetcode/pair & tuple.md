---
title: pair & tuple
---

## 如何把tuple作为unordered_set的key？
```C++
struct TupleHash {
    size_t operator()(const std::tuple<int, int, int>& t) const {
        auto [x, y, z] = t;
        // Combine hashes of the tuple elements
        return std::hash<int>()(x) ^ std::hash<int>()(y) ^ std::hash<int>()(z);
    }
};

int main() {
    // Define an unordered_set with the custom hash function
    std::unordered_set<std::tuple<int, int, int>, TupleHash> res_table;
    return 0;
}
```

