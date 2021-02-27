---
title: "C++ Concurrency in Action — Chapter 5"
excerpt: "Notes from Williams' C++ Concurrency in Action"
categories:
  - Computer Science
---

# The C++ memory model and operations on atomic types

We cover the fundamental building blocks of C++ multithreading programming, including the memory model, atomic types and operations.

## Memory model basics

### Objects and memory locations

Consider the following structure

```c++
struct Data{
  int i;							// 4 bytes
  double d;						// 8 bytes
  unsigned bf1: 10;		// 10 bits
  int bf2: 25;				// 25 bits
  int : 0;						// 0 bits
  int bf3: 99;				// 99 bits, 16 bytes after padding
  char c1, c2;				// 1 byte each
};
```

This structure takes `4+8+8+4+16=40` bytes in total, where `bf1` and `bf2` are concatenated and takes `35` consecutive bits. The zero-length bit field, which must be unnamed, separates `bf3` from `bf1` and `bf2`, causing the next field to be aligned to the next integer boundary. In this case, C++ pads another `64-35=29` bits after `bf2` so that `bf1` and `bf2` occupy `8` bytes. 

Regarding the memory location, `Data` occupies `6` memory locations. First, all objects of fundamental types occupy exactly one memory location, regardless of their size. Therefore, `i`, `d`, `c1`, and `c2` occupy `1` memory location each. Second, adjacent bit fields are part of the same memory location, regardless of their size. As a result, `bf1` and `bf2` share the same memory location, and `bf3` occupies another memory location. 

Because adjacent bit fields do not distinguish the memory location, we cannot

1. Define an array of bit fields
2. Take the address of a bit field
3. Have a reference or a pointer to a bit field

### Objects, memory locations, and concurrency

Memory location is a crucial concept for multithreading in C++; if some thread tries to *modify* the memory location which is currently accessed by some other threads, data race happens 

## Atomic operations and types in C++

### The standard atomic types

An **atomic operation** is an indivisible operation, which one can't be observed as a half done from any thread in the system.

**atomic types ** defined in `<atomic>` header may be implemented using some mutexes or other locking mechanism, rather than using lock-free atomic CPU instructions. When using atomic operations as a replacement for an operation that would otherwise uses a mutex for synchronization, the hoped-for performance gains may not be materialized if the atomic operations themselves implemented with an internal mutex. One may use member function `x.is_lock_free()` to identify, at run time, if the atomic object `x` is lock-free, or use `static constexpr` member variable `X::is_always_lock_free` at compile time.

`std::atomic_flag` is the only type that's required to be lock-free. Other atomic types may or may not be lock-free depending on the implementation. On most popular platforms, it's expected that the atomic variants of all the built-in types are indeed lock-free, but it's not required.

The standard atomic types are not copyable or copyassignable.

The standard atomic types define a set of member functions alternative the compound assignment operators where appropriate. They differ in that the member functions return the value prior to the operation while the corresponding assignments return the value stored. For example, `x += 1` returns `x` after being added by `1`, while `x.fetch_add(1)` returns the original `x` before the addition. 

#### Compare and exchange

`compare_exchange_strong(expected, desired)` and `compare_exchange_weak(expected, desired)` are used to perform [compare-and-swap](https://en.wikipedia.org/wiki/Compare-and-swap) operations. Both achieve the following logic in an atomic operation

```c++
bool compare_and_swap(T& expected, T desired) {
  if (bitwise_equal(*this, expected)) {
    *this = desired;
    return true;
  }
  expected = *this;
  return false;
}
```

Both `compare_exchange_strong()` and `compare_exchange_weak()` are often used in a loop. `compare_exchange_weak` may fail because of *spurious failure*, which happens in some platform where compare-and-swap cannot be done in a single instruction. For those platform, `compare_exchange_strong()` always check for spurious failure and mask it. This is expensive and therefore `compare_exchange_weak()` is always preferred. Although `compare_exchange_strong()` does not suffer from spurious failure, it may also fail due to concurrent write. Therefore, it's often used in a loop too.

#### User-defined type for `std::atomic<>`

`std::atomic<>` only works with user-defined types that have a *trivial* copy assignment. This means

1. There is no virtual functions or virtual base classes
2. The default compiler-generated copy assignment is available
3. `memcpy()` can be used for copy assignment

Moreover, if the user-defined type provides a custom comparison operator doing something other than the bitwise comparison, `compare_exchange_*()` will fail.

### Synchronizing operations and enforcing ordering

Let A and B be some operations. In particular, A is often an atomic write while B is an atomic read. We first introduce several terms before discussing ordering

- *Sequenced-before.* Within a same thread, A is sequenced-before B if A is evaluated before B
- *Carries-a-dependency-to.* Within a same thread, A carries a dependency to B if B reads the result of A
- *Synchronizes-with*. Between threads, A synchronizes with B if B waits for an atomic whose value is written by A in another thread
- *Dependency-ordered before*. Between threads, A is dependency-ordered before B if B reads an atomic whose value is written by A, or by some operation A is dependency-ordered before.
- *Inter-thread happens before.* Between threads, A inter-thread happens before B if 1) A synchronizes with B, 2) A is dependency-ordered before B, 3) A is sequenced-before X, X synchronizes with Y, and Y is sequenced before B 
- *Happens before.* A happens before B if 1) A is sequenced-before B, or 2) A inter-thread happens before B
- *Strongly happens before.* Strongly happens before differs from happens before only in that it excludes `memory_order_consume`.
- *Visible side effect.* The side-effect of A on a scalar M is visible to B if 1) A happens before B, and 2) there is no intermediate operation X between A and B that modifies M

There are four models for memory ordering. We order them from the mos relaxed one to the most restrictive below. It's worth noting that as the more restrictive the ordering is the more synchronization cost it incurs. 

1. **Relaxed ordering** (`memory_ordering_relaxed`). It provides no guarantee that anything that happens before A will happen before B when A in thread 1 synchronizes with B in thread 2.
2. **Release-consume ordering** (`memory_order_consume` for load) If A in thread 1 synchronizes with B in thread 2, all memory writes that happen before A become visible side effect to B and subsequent operations in thread 2 that B is dependency-ordered before.
3. **Release-acquire ordering** (`memory_ordering_acquire`, `memory_ordering_release`, `memory_ordering_acq_rel` for load, store, and read-modify-write, respectively). If A in thread 1 synchronizes with B in thread 2, all memory writes that happen before A become visible side effect to B and all subsequent operations in thread 2.
4. **Sequential consistent ordering** (`memory_ordering_seq_cst`). This is the default model for all atomic operations, which imposes the most restrictive ordering. Besides the ordering imposed by release-acquire ordering, it further impose a sequential consistent global ordering; if operation A is observed to happen before operation B in some thread, this is true for all threads. 

### Fences

If A in thread 1 synchronizes with B in thread 2 and there is a fence with `memory_ordering_release` before A and a fence with `memory_ordering_acquire` after B, then everything that happens before A becomes visible side effect to thread 2 after the acquire fence. 

## References

Williams, Anthony. 2019. *C++ Concurrency in Action, 2nd Edition*.