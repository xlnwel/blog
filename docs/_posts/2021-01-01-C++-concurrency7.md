---
title: "C++ Concurrency in Action — Chapter 7"
excerpt: "Notes from Williams' C++ Concurrency in Action"
categories:
  - Computer Science


---

# Designing lock-free concurrent data structures

## Definitions and consequences

Data structures and algorithms that use mutexes, conditional variables, and futures to synchronize the data are called *blocked* data structures and algorithms. When a thread is blocked, the OS will suspend the thread completely and allow another thread to run first.

Data structures and algorithms that don't use blocking library functions are said to be *nonblocking*. 

The reason for using a compare/exchange operation is that another thread might have modified the data in the meantime, in which case the code will need to redo part of its operation before trying the compare/exchange again.

## References

Williams, Anthony. 2019. *C++ Concurrency in Action, 2nd Edition*.

