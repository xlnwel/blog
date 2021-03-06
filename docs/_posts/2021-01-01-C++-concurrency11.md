---
title: "C++ Concurrency in Action — Chapter 10"
excerpt: "Notes from Williams' C++ Concurrency in Action"
categories:
  - Computer Science

---

# Testing and Debugging Multithreaded Application

## Types of concurrency-related bugs

Most of concurrency-related bugs falls into two categories:

- Unwanted blocking
- Race conditions

### Unwanted blocking

There are several variations on unwanted blocking

- *Deadlock*. Thread A is waiting for Thread B, which is in turn waiting for Thread A
- *Livelock*. The same as Deadlock, except that the lock is an active checking loop, such as a spin lock.
- *Blocking on I/O or other external input*. A thread is waiting for another thread, which is waiting for external input.

### Race conditions

Race conditions often cause the following types of problems:

- *Data races*. A type of race condition caused by unsynchronized concurrent access to a shared memory location. Data races usually occur through incorrect usage of atomic operations to synchronize threads or through access to shared data without locking the appropriate mutex
- *Broken invariants*. Other thread sees the intermediate state of some operations.
- *Lifetime issues*. The thread outlives the data it accesses so it is accessing data that has been deleted or otherwise destroyed.

## Techniques for locating concurrency-related bugs

### Questions to think about when reviewing multithreaded code

Here are some questions to think about when reviewing multithreaded code(I summarize them a bit, you may want to refer to Page 343 for the complete list)

- Which data needs to be protected from concurrent code?
- How do you ensure that the data is protected?
- Where in the code could other threads be at this time?
- Which mutexes does this thread hold?
- Which mutexes might other thread hold?
- Are there any ordering requirements between the operations done in this thread and those don in another? How are those requirements enforced?
- Is the data loaded by this thread still valid? Could it have been modified by other threads?
- If you assume that another thread could be modifying the data, what would that mean and how could you ensure that this never happens?

### Locating concurrency-related bugs by testing

Think about all possible situations and orderings the code might run.

### Designing for testability

In general, code is easier to test if the following factors apply:

- The responsibilities of each function and class are clear
- The functions are short and to the point
- Your test can take complete control of the environment surrounding the code being tested
- The code that performs the particular operation being tested is close together rather than spread throughout the system
- You thought about how to test the code before you wrote it

For multithreaded code, design them to be easy to be broken down into parts that communicates between threads and parts that operate within a single thread. For example, if we can divide code into multiple blocks of *read shared data/transform data/update shared data*, we can test the transform data portion using single-threaded techniques.

One thing to watch out for is that library calls that use internal variables to store state, which then become shared if multiple threads use the same set of library calls. It is easy to forget protecting these variables.

### Multithreaded testing techniques

#### Brute-force testing

Brute-force testing refers to running code many times to see if anything goes wrong. It could give false confidence sometimes. For example, on x86 and x86-64 architectures, atomic load operations are always the same, whether tagged `memory_order_relaxed` or `memory_order_seq_cst`. This means that code written using relaxed memory ordering may work on system with an x86 architecture, where it could fail on a system with a finer-grained set of memory-ordering instructions, such as SPARC.

#### Combination simulation testing

There are special softwares that can run all combination of data access using the rules of the C++ memory model and find out potential race conditions and deadlocks. But such tests are usually time-consuming and is best reserved for fine-grained tests of individual pieces of code rather than an entire application.

## References

Williams, Anthony. 2019. *C++ Concurrency in Action, 2nd Edition*.