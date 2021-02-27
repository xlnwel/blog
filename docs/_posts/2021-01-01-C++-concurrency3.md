---
title: "C++ Concurrency in Action — Chapter 3"
excerpt: "Notes from Williams' C++ Concurrency in Action"
categories:
  - Computer Science
---

# Sharing data between threads

## Problems with sharing data between threads

**race condition**: the outcome depends on the relative ordering of execution of operations on two or more threads. Problematic race conditions typically occur when an operation requires two or more separable instructions as it opens a window for other thread to step in and compromise the *data invariant*.

**data race**: a specific type of race condition that arises because the concurrent modifications to a single object. Data races cause the dreaded *undefined behavior*

### Avoiding problematic race conditions

There are several approaches to deal with problematic race condition

1. Wrap the data structure with a protection mechanism to ensure that only the thread performing a modification can see the intermediate states where the invariants are broken
2. Modify the design of the data structure and its invariants so that modifications are done as a series of indivisible changes, each of which preserves the invariants. This is generally referred to as *lock-free programming*(Chapter 7) and is difficult to get right. It requires a decent understanding of memory model (Chapter 5).
3. Record a series of reads and writes to shared memory in a log—called a *transaction*—and *commit* them in a single step. Before making a commit, we validate the memory it accessed in the past to see if another thread concurrently made changes to them. If the memory is modified, the validation fails and the transaction is either rolled back or aborted. Otherwise, the transaction is committed. This is termed *software transaction memory*, and it's still an active research area. 

## Protecting shared data with mutexes

`mutex`(mutual exclusion) is a synchronization primitive that can be used to protect shared data from being simultaneously accessed by multiple threads.

`std::lock_guard` implements an RAII idiom for a mutex; it locks the supplied mutex on construction and unlocks it on destruction

Don’t pass pointers and references to protected data outside the scope of the lock, whether by returning them from a function, storing them in externally visible memory, or passing them as arguments to user-supplied functions. (Page 43)

Some STL classes are not designed to be thread-safe; there's a potential risk of race condition when using these classes directly. For example, the following code is susceptible to data race

```c++
std::stack<std::vector<int>> data;
// ... add data
if (!data.empty()) {
	auto v = data.top();	// another thread may modify data before/after this instruction, leading to undesired behavior
	v.pop();
}
```

### Deadlock

Sometime, deadlock may not be obvious. Consider a class that defines a swap operation. In order to be thread-safe, it has to lock its own mutex as well as the other object's mutex

```c++
class X {
public:
  void swap(X& other) {
    std::lock_guard lock1(m);
    std::lock_guard lock1(other.m);
    swap(v);
  }
private:
  std::vector<int> v;
  std::mutex m;
}
```

Now if `x1.swap(x2)` and `x2.swap(x1)` are executed concurrently in two thread and each has locked its own mutex, we have deadlock. To deal with these situations, we have to lock `m` and `other.m` at once either using `std::lock`

```c++
class X {
public:
  void swap(X& other) {
    std::lock(m, other.m);
    std::lock_guard lock1(m, std::adopt_lock);
    std::lock_guard lock1(other.m, std::adopt_lock);
    swap(v);
  }
private:
  std::vector<int> v;
  std::mutex m;
}
```

Or `std::scoped_lock`

```c++
class X {
public:
  void swap(X& other) {
    std::scoped_lock lock1(m, other.m);	// relying on automic deduction of class template parameters to deduce types
    swap(v);
  }
private:
  std::vector<int> v;
  std::mutex m;
}
```

Guidelines for avoiding deadlock

- Don't wait for another thread if there's a chance it's waiting for you.
- Acquire lock in a fixed order.(Page 54)
- Use a lock hierarchy (Page 55)

### `std::unique_lock`

`std::unique_lock` usually more costly than `std::lock_guard` as it provides an option to not own a mutex.

Use `std::unique_lock` when you need to defer locking or the ownership of the lock needs to be transferred from one scope to another.

`std::unique_lock` allows to acquire and release a lock whenever it's desirable without destroying the object. It can even be passed as an argument to `std::lock`

### Locking at an appropriate granularity

Lock only when necessary. In particular, don't do any time-consuming activities such as file I/O while holding a lock.

## Alternative facilities for protecting shared data

### Protecting shared data during initialization

Sometimes data only needs to be protected during initialization and is safe for concurrent access. In that case, it's burdensome and inefficient to use a mutex for protection as one has always to check if the initialization has completed. C++ provides `std::call_once` to ensure a callable object only be called once. The necessary synchronization data is stored in a `std::once_flag` instance, which incurs a lower overhead than using a mutex explicitly. (Page 66) The following code provide a concrete example

```c++
class X {
private:
	connection_info connection_details; 
  connection_handle connection; 
  std::once_flag connection_init_flag; 
  void open_connection() { connection=connection_manager.open(connection_details); } 
public:
	X(connection_info const& connection_details_): connection_details(connection_details_) {} 
  void send_data(data_packet const& data) { 
    std::call_once(connection_init_flag, &X::open_connection, this); 	// lazy initialization
    connection.send_data(data); 
  } 
  data_packet receive_data() {
    std::call_once(connection_init_flag, &X::open_connection, this);  	// lazy initialization
    return connection.receive_data(); 
  }
};
```

It's worth noting that `std::mutex` and `std::once_flag` instances can't be copied or moved, so for classes like the above one, we have to explicitly define special member functions should you acquire them.

`static` data is guaranteed to initialized only once and therefore its initialization is thread-safe.

### Protecting rarely updated data structures

When a data structure is rarely updated, `std::mutex` presents a pessimistic option for protecting data as it prevents the possible concurrency in reading the data structure when it isn't undergoing modification. C++ provides `std::shared_mutex` and `std::shared_timed_mutex` for such cases; the latter supports additional operations (section 4.3), so the former might offer a performance benefit on some platforms.

For update operation, `std::lock_guard<std::shared_mutex>` and `std::unique_lock<std::shared_mutex>` can be used for the locking to ensure exclusive access. Those threads don't need update the data structure can use `std::shared_lock<std::shared_mutex>` to obtain shared access.

### Recursive locking

If recursively locking in a single thread is required, `std::recursive_mutex` should be used.

Most of the time, if you think you want to use a recursive mutex, you probably need to change your design instead.

## References

Williams, Anthony. 2019. *C++ Concurrency in Action, 2nd Edition*.