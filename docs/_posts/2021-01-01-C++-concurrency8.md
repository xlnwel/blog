---
title: "C++ Concurrency in Action — Chapter 8"
excerpt: "Notes from Williams' C++ Concurrency in Action"
categories:
  - Computer Science
---

# Designing concurrent code

## Techniques for dividing work between threads

`std::this_thread::yield` puts the current thread into sleep, allowing the other threads to run first.

It is not a good idea to *recursively* divide data into chunks and send them to a stack/queue for retrieval(see Listing 8.1 for an example of concurrent quick sort) as it incurs heavy contention on the access to the stack/queue. 

## Factors affecting the performance of concurrent code

One should be careful with `std::thread::hardware_concurrency` because if many applications uses it for scaling at the same time, there will be huge **oversubscription**, which incurs large context switching overhead.

### Data contention and cache ping-pong

When using the same read-modify-write atomic operations in multiple threads, contention happens as the corresponding variable may be overwritten by other threads and therefore, each processor needs to retrieve the latest value and update its cache accordingly. This is called **cache ping-pong**.

The effect of contention with mutexes are usually different from the effect of cache ping-pong because the use of a mutex naturally serializes threads at the operating system level rather than at the processor level. When there are enough threads ready to run, the operating system can schedule another to run while one thread is waiting for the mutex, whereas a processor stall prevents any threads from running on that processor.

### False sharing

Processor caches generally deal with a block of memory called *cache lines*. If the data items in a cache line is unrelated and shared by multiple threads, cache ping-pong will occur even though non of the data is shared. This is so called **false sharing**. The solution is to structure the data so that data items to be accessed by the same thread are close together in memory, whereas those to be accessed by separate threads are far apart in memory.

C++17 provides `std::hardware_destructive_interference_size` to specify the minimum offset between two objects to avoid false sharing.

### How close is your data

In a single thread, if the data is spread out in memory, it's likely it lies on separate cache lines. We call this issue **data proximity**. This can increases memory access latency and reduce performance comparing to data that's located close together.

C++17 provides `std::hardware_constructive_interference_size` to specify the maximum size of contiguous memory to promote true sharing.

## Designing data structures for multithreaded performance

### Dividing array elements for complex operations

Carefully choose the data access pattern when distributing large block of data among threads. For example, when multiplying two large $$n\times n$$ matrices $$\pmb C=\pmb A\pmb B$$, one may choose to compute $$m$$ rows in each thread. Because $$C_{ij}=\sum_kA_{ik}B_{kj}$$, this amounts to $$m \times n$$ reads from $$\pmb A$$ and $$n\times n$$ reads from $$\pmb B$$, resulting in total $$(m+n)\times n$$ reads. When $$n\gg m$$ the costs are dominated by reads from $$\pmb B$$. One better solution is to compute $$m \times m$$ sub-matrix in $$\pmb C$$ in each thread. This reduces total reads to $$2m \times n$$, significantly reducing the number of reads for $$n\gg m$$. Furthermore, it can also reduces the potential contention if there is any.

### Data access patterns in other data structures

To test if false sharing occurs when access an array of data, one may add some padding to the data structure:

```c++
struct MyData {
	data_item1 d1;
	data_item2 d2;
	char padding[std::hardware_destructive_interference_size]; 
}; 
MyData some_array[256];
```

If this improves the performance, you know false sharing was a problem, and you can either leave the padding in or work to eliminate the false sharing in another way by rearranging data accesses.

## Additional considerations when designing for concurrency

### Exception safety in parallel algorithms

Exceptions thrown in spawned threads but not handled cause the library to call `std::terminate()` to abort the application. Therefore, one should store the exception in `std::future` to allow the thread send the exception back to the main thread; the exception will be re-thrown when we call `.get()` on that `std::future`. This may be done implicitly, e.g., when an exception is thrown when calling a `std::packaged_task` object or `std::async`, the exception will automatically be stored in the corresponding `std::future`.

### Scalability and Amdahl's law

The Amdahl's law is defined as

$$
\begin{align}
P={1\over f_s+{1-f_s\over N}}
\end{align}
$$

where $$P$$ is the performance gain from using $$N$$ processors, $$f_s$$ is the fraction of the program that has to be executed serially.

Scalability is about reducing the time it takes to perform an action or increasing the amount of data that can be processed in a given time as more processors are added.

### Hiding latency with multiple threads

A thread may be blocked because, for example, it's waiting for other thread to complete some task or some I/O to complete. To make use of the processor when the thread is block we may 

1. schedule more thread so that the processor can run other thread first
2. make the thread do something else first and retrieve the result later. 

### Improving responsiveness with concurrency

Most modern graphical user interface frameworks are event-driver; the user performs actions on the user interface by pressing keys or moving the mouse, which generate a series of events or messages that the application then handles. To increase responsiveness, we usually handle the lengthy task on a new thread and leave a dedicated GUI thread to process the events. The following code shows a simple example, which only handles a single task

```c++
std::thread task_thread;
std::atomic<bool> task_cancelled(false);
void gui_thread() {
  while (true) {
    auto event = get_event();
    if (event.type == quit)
      	break;
   	process(event);
  }
}
void process(const EventType& event) {
  switch(event.type) {
  case start_task:
    task_cancelled = false;
    task_thread = std::thread(task);
    break;
  case stop_task:
    task_cancelled = true;
    task_thread.join();
    break;
  case task_complete:
    task_thread.join();
    display_results();
    break;
  default:
    //...  
  }
}
```

## References

Williams, Anthony. 2019. *C++ Concurrency in Action, 2nd Edition*.

