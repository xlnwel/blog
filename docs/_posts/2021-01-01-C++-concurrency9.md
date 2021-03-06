---
title: "C++ Concurrency in Action — Chapter 9"
excerpt: "Notes from Williams' C++ Concurrency in Action"
categories:
  - Computer Science
---

# Designing advanced thread management

## Thread pools

### Avoiding contention on the work queue

Sharing a single work queue/stack between threads increases contention and cache ping-pong. One way to avoid such overheads is to use a separate queue per thread. This can be achieved with resort to `thread_local` specifier, which ensures that the object has thread storage duration. 

## Interrupting threads

C++20 introduces `std::jthread` for stoppable threads. `std::jthread` can be stopped by calling `.request_stop()`. Furthermore, `.request_stop()` is automatically invoked when `std::jthread` is destroyed if the task has not finished.

### Launching and interrupting another thread

One way to interrupt another thread is to define an atomic Boolean variable and have that thread constantly check this variable. If it is true, throw an exception indicating that the thread is interrupt.

## References

Williams, Anthony. 2019. *C++ Concurrency in Action, 2nd Edition*.

