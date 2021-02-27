---
title: "C++ Concurrency in Action — Chapter 6"
excerpt: "Notes from Williams' C++ Concurrency in Action"
categories:
  - Computer Science

---

# Designing lock-based concurrent data structures

The point is to break down the data structure into a more fine-grained one so that we can apply separate locks to different parts of the data structure. This can significantly increase the concurrency but meanwhile it typically requires more careful considerations on the design. Here's a list of questions to help consider the data structure for concurrency

- Can the scope of locks be restricted to allow some parts of an operation to be performed outside the lock?
- Can different parts of the data structure be protected with different mutexes?
- Do all operations require the same level of protection?
- Can a simple change to the dat structure improve the opportunities for concurrency without affecting the operational semantics

Here's some guidelines for thread safe code.

- Ensure that no thread can see a state where the invariants of the data structure have been broken by the actions of another thread.
- Take care to avoid race conditions inherent in the interface to the data structure by providing functions for complete operations rather than for operation steps. (e.g. combining `empty`, `top` and `pop` in a single operation)
- Pay attention to how the data structure behaves in the presence of exceptions to ensure that the invariants are not broken. More often than not, RAII helps
- Minimize the opportunities for deadlock when using the data structure by restricting the scope of locks and avoiding nested locks where possible.

## Lock-based concurrent data structures

In this section, Anthony shows an example of thread-safe queue, starting from the primitive one that adds lock to the `std::queue`. Then he moves to one using list nodes which allows us to apply separate `mutex`es to the `head` and `tail` node pointers. Moreover, he argues the benefit of using a dummy `head`, which frees us from checking if the `head` is `nullptr` and thus avoids locking `head_mutex` in the `push` operation. An interesting practice of the final thread-safe queue is that it always stores data in the current tail and update the tail thereafter, see the code below

```c++
void push(T data) {
  {
    auto p = std::make_unique<Node>();
    std::lock_guard l(tail_mutex);
    tail->data = std::make_unique<T>(std::move(data));   // we add data to the current tail, this allows us to move head to the next when popping
    tail->next = std::move(p);
    tail = tail->next.get();
  }
  data_cond.notify_one();
}
```

This ensures that `tail` is always valid after a `pop` operation

```c++
template<typename T>
T ThreadSafeQueue<T, std::list<T>>::pop() {
  std::unique_lock l(head_mutex);
  data_cond.wait(l, [this] { return head.get() != get_tail(); });
  auto data = std::move(*head->data);
  head = std::move(head->next);		// we move head to the next so that the tail is always valid
  return data;
}
```

## Designing more complex lock-based data structures

In this section, Anthony designs a thread-safe look-up table step by step. He first associates each bucket in the hash table with a mutex. Then he uses a custom list as the bucket type, which allows a per-node lock instead of a lock for the whole bucket. 

## References

Williams, Anthony. 2019. *C++ Concurrency in Action, 2nd Edition*.

