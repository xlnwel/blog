---
title: "C++ Concurrency in Action — Chapter 1"
excerpt: "Notes from Williams' C++ Concurrency in Action"
categories:
  - Computer Science
---

# Hello, word of concurrency in C++

## Concurrency with Multiple Processes

Advantage:

1. Some systems provide higher-level communication mechanisms so that it can be easier to write safe concurrent code with process rather than threads.
2. You can run the separate processes on distinct machines connected over a network.

Downside:

1. Communication between processes is often either complicated to set up or slow, or both, because operating systems typically provide a lot of protection between processes to avoid one process accidentally modifying data belonging to another process.
2. There is an inherent overhead in running multiple processes: it takes time to start a process, the operating system must devote internal resources to managing the process, and so forth.

## Concurrency with Multiple Threads

All threads in a process share the same address space, and most of the data can be accessed directly from all threads—global variables remain global, and pointers or references to objects or data can be passed around among threads. Although it’s often possible to share memory among processes, this is complicated to set up and often hard to manage, because memory addresses of the same data aren’t necessarily the same in different processes.

The flexibility of shared memory also comes with a price: if data is accessed by multiple threads, the application programmer must ensure that the view of data seen by each thread is consistent whenever it’s accessed.

## When use concurrency?

Two main reasons to use concurrency in an application: separation of concerns and performance.

There are two ways to use concurrency for performance.

1. Task parallelism. Divide a single task into parts and run each in parallel. There are two variants regarding where the division happens: in terms of processing, one thread performs one part of the algorithm while another thread performs a different part; or in terms of data, one thread performs the same operation on different parts of the data. The latter is called *data parallelism*. Algorithms that are readily scale to such parallelism are called *embarrassingly parallel*, *naturally parallel*, or *conveniently concurrent*.
2. Use available parallelism to solve bigger problems; rather than processing one file at a time, process 2 or 10, as appropriate.

## When not use concurrency?

Simply put, when the cost does not match the gain. The cost comes from three places

1. *Intellectual cost*: code using concurrency is often hard to understand
2. *Thread overhead:* the performance gain might not be as large as expected; there’s an inherent overhead associated with launching a thread, because the OS has to allocate the associated kernel resources and stack space and then add the new thread to the scheduler, all of which takes time.

## References

Williams, Anthony. 2019. *C++ Concurrency in Action, 2nd Edition*.