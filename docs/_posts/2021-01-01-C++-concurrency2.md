---
title: "C++ Concurrency in Action — Chapter 2"
excerpt: "Notes from Williams' C++ Concurrency in Action"
categories:
  - Computer Science
---

# Managing threads

## Basic thread management

### Launching a thread

```c++
std::thread new_thread(callable);
```

where `callable` could be any callable object, such as function or a class with a function call operator. 

When `callable` is a callable class, the function object is copied into the storage belonging to the new thread and invoked from there. There are two implications: 1) it is essential that the copy behaves equivalently to the original, or the result may not be what's expected. 2) the internal state of `callable` is copied and if there is no pointer or reference to some local variable, calling functions that use the internal state is thread-safe even when the original `callable` is destroyed

Don't try to be smart to write some code like

```c++
std::thread new_thread(BackgroundTask());
```

Where `BackgroundTask` is a class with a function call operator. `BackgroundTask()` here will be interpreted as a function pointer that takes no parameters and returns a `BackgroundTask` object. Consequently, `new_thread` will be interpreted as a function declaration that takes a single parameter `BackgroundTask()` and returns a `std::thread` object. This can be avoided by one of the following approach

```c++
std::thread new_thread((BackgroundTask()))	// parentheses prevent interpretation as a function declaration
std::thread new_thread{BackgroundTask()}
```

Once launching a thread, you have to decide, before the thread object is destroyed, whether to wait for the thread to finish(*join*) or leave it to run on its own(*detach*). Otherwise the program is terminated(the `std::thread` destructor calls `std::terminate()`). If you detach the thread, you need to ensure that the data accessed by the thread is valid until the thread has finished with it. 

### Joining a thread

Be aware that `join()` might be skipped when an exception is thrown in the new thread, when you defer `join()`. You can ensure a thread be joined using `try...catch` to join the thread when an exception occurs. Alternatively, you can use the RAII idiom as follows

```c++
class ScopedThread {
public:
  explicit ScopedThread(std::thread&& tt): t(std::move(tt)) {}
  ScopedThread(const ScopedThread&)=delete;
  ScopedThread& operator=(const ScopedThread&)=delete
  ~ScopedThread() {
    if (t.joinable())
      	t.join();
  }
private:
  std::thread t;
}
```

Note that we explicitly delete copy and assignment operations to avoid unintended move operations. This is more imperative when `ScopedThread` only keep a reference to a local thread object, in which allowing copy and assignment operations may allow `ScopedThread` outlive the scope of the thread it refers to.

### Detaching a thread

If you decide to detach a thread, do it immediately after launching a thread.

When a thread is detached, the ownership and control are passed over to the C++ Runtime Library, which ensures the resources associated with the thread are correctly reclaimed when the thread exits.

Detached threads are often called *daemon threads*.

Two typical cases of detached threads: 1) threads run for almost the entire lifetime of the application, performing a background task such as monitoring the filesystem, learning ensued entries out of object caches, or optimizing data structures. 2) when there's another mechanism for identifying when the thread has completed or where the thread is used for a fire-and-forget task.

An example of detached thread in Page 23 shows that a word processor launches a detached thread for every new document window.

## Passing arguments to a thread function

By default, arguments are copied into internal storage, where they can be accessed by the thread and passed to the callable object or function as *rvalue* as if they were temporaries. This is done even if the corresponding parameter in the function is expecting a reference.

Extra care must be taken when the argument is a pointer. For example,

```c++
void f(int i,std::string const& s); 
void oops(int some_param) {
	char buffer[1024]; 
  sprintf(buffer, "%i", some_param);
  std::thread t(f, 3, buffer);
  t.detach();
}
```

In this example, it's the pointer to `buffer` being passed through the new thread and there's a significant chance that `oops` function will exit before the buffer has been converted to a `std::string` on the new thread—the internal storage only stores a pointer `char*`—thus leading to undefined behavior. The solution is to cast to `std::string` before passing the buffer to the `std::thread` function

```c++
void f(int i,std::string const& s); 
void oops(int some_param) {
	char buffer[1024]; 
  sprintf(buffer, "%i", some_param);
  std::thread t(f, 3, std::string(buffer));
  t.detach();
}
```

Note that if the function accepts plain references to non-const data, it will fail to compile as such references cannot bind to rvalue. The solution is to wrap the arguments that need to be references in `std::ref`.  (Page 25)

```c++
void f(vector<int>&);
int main() {
  vector<int> data;
  std::thread t(f, data);						// error as f expects a plain reference
  std::thread t(f, std::ref(data));	// okay
}
```

This parameter-passing semantics is defined in the same way as `std::bind`. This means that, for example, you can pass a member function pointer as the function, provided you supply a suitable object pointer as the first argument (Page 26)

```c++
class X{
public:
  void f();
}
X x;
std::thread t(&X::f, &x);
```

## Identifying threads

We can get thread id from its associated `std::thread` object by calling `get_id()` member function. `get_id()` returns a default-constructed `std::thread::id` object, which indicates "not any thread". Alternatively, the identifier for the current thread can be obtained by calling `std::this_thread::get_id()`.

## References

Williams, Anthony. 2019. *C++ Concurrency in Action, 2nd Edition*.