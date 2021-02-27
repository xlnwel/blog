---
title: "C++ Concurrency in Action — Chapter 10"
excerpt: "Notes from Williams' C++ Concurrency in Action"
categories:
  - Computer Science
---

# Parallel algorithms

### Parallelizing the standard library algorithms

C++17 introduces parallel algorithms to STL. These are additional overloads of their single-thread versions, such as `std::sort`, `std::transform`. Parallel algorithms have the same signature as the single-thread versions, except for the additional of a new first parameter, which specifies the *execution policy* to use.

### Execution policies

The standard specifies four execution policies:

- `std::execution::sequenced_policy`, which forces the algorithm to execute on this thread. Different from the one without execution policy, the order of this operation is not guaranteed. For example, the following code may store numbers 1 to 100 in any order in `v`

  ```c++
  std::vector<int> v(100);
  int i = 0;
  std::for_each(std::execution:seq, v.begin(), v.end(), [&i](int &x){ x = ++i; })
  ```

- `std:execution::parallel_policy`, which indicate that the algorithm execution's may be parallelized. Any potential data race would result in undefined behavior, even the algorithm is actually executed on this thread only. 

- `std::execution::parallel_unsequenced_policy`, which indicates that the algorithm's execution may be parallelized, vectorized, or migrate across threads.

- `std::execution::unsequenced_policy`, which indicates that the algorithm's execution maybe vectorized. Any locks may invalidate vectorization, resulting in undefined behavior.

Note that all these policies are *permissions*, not *requirements*, meaning that the algorithm may choose to ignore the policy if it wishes. The more relaxed execution gives the library more freedom to improve the performance, meanwhile it puts more tighter requirements on the code.

You cannot rely on being able to construct objects from these policy class yourself because they might have special instantiation requirement. Instead, we have to copy the following three corresponding policy objects defined in the `<execution>` header

- `std::execution::seq`
- `std::execution::par`
- `std::execution::par_unseq`

#### Exception behavior

Algorithms with a specified execution policy will call `std::terminate` if there are any unhandled exceptions, except for `std::bad_alloc`, which is thrown if the library cannot obtain sufficient memory resources for its internal operations 

## References

Williams, Anthony. 2019. *C++ Concurrency in Action, 2nd Edition*.

