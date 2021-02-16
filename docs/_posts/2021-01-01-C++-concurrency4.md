---
title: "C++ Concurrency in Action — Chapter 4"
excerpt: "Notes from Williams' C++ Concurrency in Action"
categories:
  - Computer Science
---

## Synchronizing concurrent operations

## Waiting for an event or other condition

`std::condition_variable` is used to block a thread until another thread modifies a shared variable and notifies(via `.notify_one`) the `conditional_variable`. `std::condition_variable` only works with `std::unique_lock<std::mutex>`. To work with other locks, use `std::condition_variable_any` instead.

The following example shows how to use `condition_variable`

```c++
std::mutext m;
std::queue<data_type> dq
std::condition_variable cond;
void generate_data() {
  while (has_data()) {
    auto data = prepare_data();
    {
      std::lock_guard l(m);
      dq.push(data)
    }
    cond.notify_one();	// notifies waiting thread(s)
  }
}
void process_data() {
  while(true) {
    std::unique_lock l(m);
    cond.wait(lk, []{return !dq.empty();});	// unlock, until receiving notification from another thread.
    auto data = dq.front();
    dq.pop();
    l.unlock();
    process_data(data);
  }
}
```

Note that the waiting thread may wake multiple times and the predicate (`[]{return !dq.empty();}` in the above example) may be invoked multiple times depending on the underlying implementation. Therefore, it is advisable to not use a function with side effects as the predicate.

When several threads are waiting for the same event, and all of them need to respond, use `.notify_all` to notify all threads.

## Waiting for one-off events with futures

`std::future` is *ready* when the associated event sends its result to the future or finished unexpectedly. 

`std::future` is only *moveable* and can only call `.get()` *once* to retrieve the value.

`std::shared_future` is *copyable* and can call `.get()` multiple times.

### Returning values from background tasks

If the waiting thread is going to wait only once, a conditional variable might not be the best choice of synchronization mechanism. This is where futures come into play. 

When we want to execute some task and retrieve the return latter, we can use `std::async(callable)`, which runs `callable` and returns a `future` object. By default, the launch policy is `std::launch::async | std::launch::deferred`, which means that `std::async` may or may not run `callable` asynchronously depending the library implementation. One can explicitly specify policy to `std::launch::async` to force `callable` into running a different thread.

### Associating a task with a future

`std::packaged_task<>` wraps an callable so that it can be invoked asynchronously or synchronously depending on where the task is called. Different from `std::async`, `std::packaged_task` does not starts a new thread on its own. Instead, it returns a task object, which can be passed to another thread and invoked from there. The original thread may hold a `feature` from `std::packaged_task<>::get_future()` so that it can retrieve the result of `callable` whenever appropriate.

```c++
ThreadSafeQueue<std::packaged_task<int()>> task_queue;	// thread safe queue, which handles locks inside
void task_execution_thread() {
  bool x = true;
	while (x) {	// for debugging purpose, we only execute this loop once
    auto task = task_queue.pop();	// Returns the front task and removes it from queue. Waits if task_queue is empty
    task();	// execute task
    x = false;
  }
}

template<typename ReturnType, typename... Args>
std::future<ReturnType> post_task(std::function<ReturnType(Args...)> f) {
  std::packaged_task<ReturnType(Args...)> task(f);
  std::future res = task.get_future();
  task_queue.push(std::move(task));   // packaged_task is not copyable
  return res;
}
```

### Making (std::)promises

`std::promise` provides a facility to store a value(e.g., `.set_value()`) or an exception(e.g., `.set_exception`) that is later acquired asynchronously via a `std::future` object created by the `std::promise` object.

```c++
void working_thread(std::promise<bool> p) {
  std::cout << "do some work\n";
  p.set_value(true);
}
int main() {
  std::promise<bool> done_promise;
  auto done_future = done_promise.get_future();
  std::thread t(working_thread, std::move(done_promise));
  done_future.get();
  t.join();
}
```

### Saving an exception for the future

`std::promise` provides a way to store exception through `.set_exception()`. One may set exception in the `try...catch` block when an exception is throw:

```c++
try {
  throw SomeException();
}
catch(...) {
  some_promise.set_exception(std::current_exception())
}
```

Or store a new exception without throwing

```c++
some_promise.set_exception(std::make_exception_ptr(SomeException()));
```

The latter should be preferred if the type of the exception is known; not only does it simplifies code, but it also provides the compiler with greater opportunity to optimize the code.

A future stores an exception when `std::promise` and `std::packaged_task` associated with the future are destroyed without calling either the `set_*` functions or invoking the packaged task. 

### Waiting from multiple threads

Accessing a single `std::share_future` object from multiple threads is not safe and requires protection mechanism such as locks. It's better to pass a *copy* of the `std::shared_future` object to each thread, so each thread can access its own local `std::shared_future` object safely, as the internals are now correctly synchronized by the library.

`std::shared_future` instances are usually constructed from

1. `std::future` instance via move constructor, which transfers the ownership of the synchronous state from `std::future` to `std::shared_future`: `std::shared_future sf = std::future;`
2. `.get_future()` member function of `std::promise` and `std::packaged_task` as the transfer of ownership is implicit for rvalues: `std::shared_future sf = some_promise.get_future()'`
3. `.share()` member function of `std::future`: `auto sf = some_future.share()`

## Waiting with a time limit

## Using synchronization of operations to simplify code

## References

Williams, Anthony. 2019. *C++ Concurrency in Action, 2nd Edition*.