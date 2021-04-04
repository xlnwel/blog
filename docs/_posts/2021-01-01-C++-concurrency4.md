---
title: "C++ Concurrency in Action — Chapter 4"
excerpt: "Notes from Williams' C++ Concurrency in Action"
categories:
  - Computer Science
---

# Synchronizing concurrent operations

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
2. `.get_future()` member function of `std::promise` and `std::packaged_task` as the transfer of ownership is implicit for rvalues: `std::shared_future sf = some_promise.get_future();`
3. `.share()` member function of `std::future`: `auto sf = some_future.share()`

## Waiting with a time limit

### Clocks

Specifically, a clock is a class that provides four distinct pieces of information:

- The time now
- The type of the value used to represent the times obtained from the clock
- The tick period of the clock
- Whether or not the clock ticks at a uniform rate and is therefore considered to be a steady clock

### Durations

`std::chrono::duration<>` specifies a time interval. It can be used with `.*_for()` member functions.

### Time points

`std::chrono::time_point<>` represents a point in time. It's the return type of `std::chrono::system_clock::now` and can be used with `.*_until()` member functions.

### Functions that accept timeouts

| Class/Namespace                                              | Functions                                      | Return Values                                                |
| ------------------------------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------ |
| `std::this_thread`                                           | `sleep_for()`, `sleep_until`                   | N/A                                                          |
| `std::condition_variable`, `std::condition_variable_any`     | `wait_for`, `wait_until`                       | `std::cv_status::timeout`, `std::cv_status::no_timeout`      |
| `std::timed_mutex`, `std::recursive_timed_mutex`,`std::shared_timed_mutex` | `try_lock_for`, `try_lock_until`               | `bool`                                                       |
| `std::shared_timed_mutex`                                    | `try_lock_shared_for`, `try_lock_shared_until` | `bool`                                                       |
| `std::unique_lock<timed_mutex>`, `std::shared_lock<shared_timed_mutex>` | `try_lock_for`, `try_lock_until`               | `bool`                                                       |
| `std::future`, `std::shared_future`                          | `wait_for`, `wait_until`                       | `std::future_status::timeout`, `std::future_status::ready`, `std::future_status::deferred` |

## Using synchronization of operations to simplify code

### Synchronizing operations with message passing

Communicating Sequential Processes have no shared data; all communication is passed through the message queues. Each thread is therefore a state machine: when it receives a message, it updates its state and maybe sends one or more message to other threads. Note that "state" here is not necessarily some variable; it can be some function that instructs what to do next. For example, the ATM example provided in Section 4.4.2 treats a member function as the state of the logic thread; the logic thread repeatedly calls the state and may switch the state based on the new message received. 

### Continuation-style concurrency with the Concurrency TS

`std::experimental::future` has a member function `then`, which spawns a new thread to do following-up tasks when the current `std::experiment::future` is *ready*. The following-up task should be a function that takes as input a future of previous return(e.g., `std::future<int>` if the previous task returns a `int`. Note that it's `std::future` rather than `std::experimental::future`). `std::experimental::future` is not compatible with `std::future` and is obtained from the corresponding functions in `std::experimental`, such as `std::experimental::promise`.

### Waiting for more than one future

When there's a series of tasks running in parallel and you want to retrieve them when they are all done and do some following-up tasks, one may try to call `.get()` member function of the corresponding futures one by one either synchronously in the current thread or asynchronously in a new thread as follows.

```c++
std::async([all_results=std::move(task_futures)] {
  std::vector<DataType> v;
  for (auto& f: all_results) {
    v.push_back(f.get());
  }
  return do_some_thing(v);
})
```

Either way, there will be a thread waiting for each task and repeatedly being woken up as each result become available. Not only does this occupy the thread doing the waiting, bu it adds additional context switches as each future becomes ready.

With `std::experimental::when_all`, this waiting and switching can be avoided. It accepts a set of futures to be waited on and returns a new future that becomes ready when all the futures are ready. The following code demonstrates an example

```c++
std::experimental::when_all(task_futures.begin(), task_futures.end())
	.then([](auto f) {// we use auto to deduce the type: std::future<std::vector<std::experimental::future<DataType>>>
  std::vector ready_futures = f.get();
  std::vector<DataType> v;
  for (auto& f: all_results) {
    v.push_back(f.get());	// will not block
  }
  return do_some_thing(v);
})
```

### Waiting for the first future in a set with `when_any`

`std::experimental::when_any` creates a future that becomes ready when at least one of the input futures become ready. It returns a structure `when_any_result<>`, which contains two members: 

1. `futures` contains all input futures
2. `index` is the index of ready future

An example of retrieving the ready future is given below

```c++
std::experimental::future<std::experimental::when_any_result<
  std::vector<std::expeirmental::future<DataType>>> result_of_when_any;
auto results = result_of_when_any.get();
DataType = results.futures[results.index].get();	// retrieve data
```

### `std::latch`

A **latch** is a synchronization object that becomes ready when its counter is decremented to zero. It's *useful when you are waiting for a set of threads to reach a particular point in code*.

```c++
const thread_count = max(std::thread::hardware_concurrency(), 2);
std::latch done(thread_count);
DataType data[thread_count];
std::vector<std::future<void>> threads;
for (int i = 0; i < thread_count; ++i) {
  threads.push_back(std::async(std::launch::async, [&, i]{
    data[i] = make_data(i);
    done.count_down();	// counts down the latch
    do_more_stuff();
  }));
}
done.wait();	// waits on the latch
process_data(data);	// processes data. Threads may not be completed
```

### `std::barrier`

A barrier is a *reusable* synchronization component used for *internal synchronization between a set of threads*. When threads arrive at the barrier(at the point of calling `.arrive_and_wait()`), they block until all of the threads involved have arrived at the barrier, at which point they are all released.

```c++
// data objects
DataSource source;
DataSinc sink;
std::vector<DataChunk> chunks;
std::vector<DataChunk> results;

constexpr thread_count = max(std::thread::hardware_concurrency(), 2);
std::barrier sync(thread_count);
std::vector<std::future<void>> threads;
for (int i = 0; i < thread_count; ++i) {
  threads.push_back(std::async(std::launch::async, [&, i]{
    while (!source.done()) {
      if (!i) {
        chunks = source.get_data();	// get data in thread with i == 0
      }
      sync.arrive_and_wait();	// all threads wait for chunks to be ready
      results[i] = process(chunks[i]);
      sync.arrive_and_wait();	// waits for all threads to finish their processing
      if (!i) {
        sink.write_data(results)
      }
    }
  }));
}
```

### `std::flex_barrier`

`std::experimental::flex_barrier` add a completion function object to the completion phase where all threads arrive at the barrier. The completion function is run on one thread after all threads have arrived and returns a number indicating the number of participating threads in the next cycle (\\(-1\\) indicates the set of participating threads is unchanged).

```c++
// data objects
DataSource source;
DataSinc sink;
std::vector<DataChunk> chunks;
std::vector<DataChunk> results;

auto split_source = [&] {
  if (!source.done) {
    chunks = source.get_data();
  }
}
split_source();	// prepare chunks

constexpr thread_count = max(std::thread::hardware_concurrency(), 2);
std::flex_barrier sync(thread_count, [&] {
  sink.write_data(results);
  split_source();
  return -1;	// the number of participating threads remains unchanged
});
std::vector<std::future<void>> threads;
for (int i = 0; i < thread_count; ++i) {
  threads.push_back(std::async(std::launch::async, [&, i]{
    while (!source.done()) {
      results[i] = process(chunks[i]);
      sync.arrive_and_wait();	// waits for all threads to finish their processing, and 
      }
    }
  }));
}
```

## References

Williams, Anthony. 2019. *C++ Concurrency in Action, 2nd Edition*.