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

The reason for using a compare/exchange operation is that another thread might have modified the data in the meantime, in which case the code will need to redo part of its operation(e.g., the false part of compare/exchange) before trying the compare/exchange again.

## Detecting nodes that can't be reclaimed using hazard pointers

Hazard pointers refers to a technique discovered by Maged Michael. The idea is that if a thread is going to access an object that would be potentially deleted by another thread, it first set a hazard pointer to reference the object, informing the other thread that deleting the object would indeed be hazardous. Once the object is no longer needed, the hazard pointer is cleared. Because a hazard node is always set before reading, reading is always safe.

It's easy to see that hazard pointers consists of two parts:

1. The thread trying to access an object sets the hazard pointer first and then accesses it.
2. The thread trying to delete an object first check if any hazard pointers references the object. If there is one, it defers the reclamation; otherwise, it deletes it.

We demonstrate hazard pointers with an implementation of `pop` operation below. Note that a multi-thread `stack` does not need such a complicated `pop`—one is free to delete `old_head` after retrieving data as long as the `stack` does not define other operations that access the data.

```c++
std::shared_ptr<T> pop() {
  std::atomic<void*>& hp = get_hazard_pointer_for_current_thread();
  Node* old_head = head.load();	// head is an atomic /of void*
  do {
    Node* tmp;
    // Loop until both old_head and hp points to head
    // this eliminates the problem introduced by other 
    // threads changing the head after setting the hazard pointer
    do {
      tmp = old_head;
      hp.store(old_head);
      old_head = head.load();
    }	while (old_head != head);
  // change the head to the next after the hazard pointer is set
  // compare_exchange further ensures that the hazard point and 
  // old_head point to the head we're gonna delete. The strong
  // version is used to avoid resetting hp because of spurious failures
  } while (old_head && !head.compare_exchange_strong(old_head, old_head->next));
  hp.store(nullptr);	// clear the hazard pointer as we've claimed the old_head as ours
	std::shared_ptr<T> res;
  if (old_head) {
    res.swap(old_head->data);	// it's free to swap data as no read threads try to access data
    if (outstanding_hazard_pointers_for(old_head))
      reclaim_later(old_head);
    else
      delete old_head;
    delete_nodes_with_no_hazard(); // delete nodes that are previously stored by reclaim_later
  }
  return res;
}
```

We don't elaborate functions used here. Instead, we only outline the utility of each function below. For interested readers, please refer to section 7.2.2 of [Williams 2019](#ref1) for the complete code. 

- `get_hazard_pointer_for_current_thread()`: returns a hazard pointer for the current thread. This hazard pointer is a global `atomic` but is currently uniquely owned by the current thread. This means that other threads may check it's value but is not allowed to write to it.
- `outstanding_hazard_pointers_for(old_head)`: check all hazard pointers and return if there is any hazard pointer pointing to `old_head`.
- `reclaim_later(old_head)`: add `old_head` to a `reclaim_list` so that we can reclaim it later
- `delete_nodes_with_no_hazard()`: check the `reclaim_list` to delete a node if any there is no hazard pointer associated to it.

## Detecting nodes in use with reference counting

Another technique to prevent from dereferencing a delete object is to check how many pointers are currently pointing to the object before deleting. This idea is exactly the same as `shared_ptr`. As C++20 officially introduces `std::atomic<std::shared_ptr<T>>`, this method becomes trivial(for previous C++ standards, [standalone functions](https://en.cppreference.com/w/cpp/memory/shared_ptr/atomic), deprecated as of C++20, may be used for atomic access to `std::shared_ptr`). We only briefly discuss the data structure below

```c++
struct Node;
struct CountedNodePtr {
  int external_count;
  Node* ptr;
};

struct Node {
  std::shared_ptr<T> data;
  std::atomic<int> internal_count;
  CountedNodePtr next;
};
std::atomic<CountedNodePtr> head;
```

Here, we use `CountedNodePtr` instead of `Node*` as the atomic type, which includes an additional counter `external_count`. `external_count` increases by 1 when any thread tries to read the pointer. When a thread finishes its reading, we decrease `internal_count` by 1. When `external_count` and `internal_count` sum to 0, we delete the node. The utility of the separation of `external_count` from `internal_count` is twofold: 1) `external_count` provides a way to inform other threads that I'm gonna to access `ptr`, don't delete it until I finish. 2) `internal_count` counts how many accesses are finished. One may attempt to remove `internal_count` and decrease `external_count` instead. However, this cannot be done without changing the data structure as `external_count` is a local object and will be discarded once the corresponding `CountedNodePtr` is destroyed. We refer interested reader to subsection 7.2.4 of [Williams 2019](#ref1) for detailed operations of this method.

## Guidelines for working lock-free data structures

Several guidelines for working lock-free data structures are listed below

- Use `std::memory_order_seq` for prototyping. 
- Use a lock-free memory reclamation scheme
- Watch out for the ABA problem. Extra care must be taken when reusing nodes. For example, compare/exchange may happen far later after the `expected` value `A` is stored. During that period, the `atomic` may have changed to some other value `B` and changed back to the `expected` value `A`. In that case, compare/exchange still succeed as both the `atomic` and `expected` has value `A`, but they are no longer the same `A` and in most cases, were not supposed to be treated as if they are.
- Identify busy-wait loops and help the other thread (see subsection 7.2.6 for an example)

## References

<a name='ref1'></a>Williams, Anthony. 2019. *C++ Concurrency in Action, 2nd Edition*.

