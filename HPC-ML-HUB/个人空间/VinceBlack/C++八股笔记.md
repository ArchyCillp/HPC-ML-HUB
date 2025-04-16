直接看这个网站里面的就行
https://github-pages.ucl.ac.uk/research-computing-with-cpp/02cpp1/sec05Pointers.html#:~:text=Raw%20pointers%20can%20be%20used,t%20point%20to%20invalid%20memory.

# General guidelines
Functions exposed via the C++ API must be stateless. Things that are OK to be exposed on the interface:

1. Any [POD](https://en.wikipedia.org/wiki/Passive_data_structure) - see [std::is_pod](https://en.cppreference.com/w/cpp/types/is_pod) as a reference for C++11 POD types.
    
2. `raft::resources` - since it stores resource-related state which has nothing to do with model/algo state.
    
3. Avoid using pointers to POD types (explicitly putting it out, even though it can be considered as a POD) and pass the structures by reference instead. Internal to the C++ API, these stateless functions are free to use their own temporary classes, as long as they are not exposed on the interface.
    
4. Accept single- (`raft::span`) and multi-dimensional views (`raft::mdspan`) and validate their metadata wherever possible.
    
5. Prefer `std::optional` for any optional arguments (e.g. do not accept `nullptr`)
    
6. All public APIs should be lightweight wrappers around calls to private APIs inside the `detail` namespace.

## Explicit template instantiation
https://stackoverflow.com/questions/2351148/explicit-template-instantiation-when-is-it-used
used in cuvs.




