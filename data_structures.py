"""
MSCS-532 Assignment 6, Part 2: Elementary Data Structures

Arrays, matrices, stacks, queues, singly linked lists, and rooted trees,
implemented from scratch.

Run:  python data_structures.py      (runs the correctness suite)

A note on what "from scratch" means here
----------------------------------------
Python's built-in list is already a dynamic array, so building a dynamic
array by calling list.append would be assuming the thing being implemented.
Instead DynamicArray allocates a fixed-size backing store of None slots and
manages capacity, element count, index arithmetic, growth, and shrinkage
itself. Every other array-backed structure in this file is built on top of
that, and the linked structures allocate their own nodes. The only Python
facilities used are raw slot assignment and allocation of a block of slots.
"""


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------
class DynamicArray:
    """
    A growable array over a fixed-capacity backing store.

    Operation             Time              Why
    --------------------- ----------------- -----------------------------
    access(i)             O(1)              one address computation
    set(i, v)             O(1)              same
    append(v)             O(1) amortized    doubling spreads the copy cost
    insert(i, v)          O(n)              shift the tail right
    delete(i)             O(n)              shift the tail left
    pop()                 O(1) amortized    no shifting needed
    search(v)             O(n)              no ordering to exploit

    Growth policy: double on overflow, halve when the array drops to a
    quarter full. The quarter-full trigger rather than half-full is what
    keeps the amortized cost O(1); shrinking at half-full lets an
    alternating append/pop sequence at the boundary resize on every
    operation.
    """

    def __init__(self, capacity=8):
        self._capacity = max(1, capacity)
        self._store = [None] * self._capacity
        self._n = 0
        self.resizes = 0          # instrumentation for the report

    def __len__(self):
        return self._n

    @property
    def capacity(self):
        return self._capacity

    def _resize(self, new_capacity):
        """Allocate a new block and copy the live elements across. O(n)."""
        new_store = [None] * new_capacity
        for i in range(self._n):
            new_store[i] = self._store[i]
        self._store = new_store
        self._capacity = new_capacity
        self.resizes += 1

    def _check(self, i):
        if not 0 <= i < self._n:
            raise IndexError(f"index {i} out of range for length {self._n}")

    def access(self, i):
        """O(1). The defining property of an array."""
        self._check(i)
        return self._store[i]

    def set(self, i, value):
        """O(1)."""
        self._check(i)
        self._store[i] = value

    def append(self, value):
        """O(1) amortized, O(n) on the resize itself."""
        if self._n == self._capacity:
            self._resize(self._capacity * 2)
        self._store[self._n] = value
        self._n += 1

    def insert(self, i, value):
        """
        O(n). Everything from index i onward moves one slot right, so the
        cost is proportional to how much of the array sits after i.
        Inserting at the front is the worst case: n moves.
        """
        if not 0 <= i <= self._n:
            raise IndexError(f"index {i} out of range for insert")
        if self._n == self._capacity:
            self._resize(self._capacity * 2)
        for j in range(self._n, i, -1):
            self._store[j] = self._store[j - 1]
        self._store[i] = value
        self._n += 1

    def delete(self, i):
        """O(n). The hole at i has to be closed by shifting the tail left."""
        self._check(i)
        removed = self._store[i]
        for j in range(i, self._n - 1):
            self._store[j] = self._store[j + 1]
        self._store[self._n - 1] = None
        self._n -= 1
        if self._capacity > 8 and self._n <= self._capacity // 4:
            self._resize(max(8, self._capacity // 2))
        return removed

    def pop(self):
        """O(1) amortized. Removing the last element shifts nothing."""
        if self._n == 0:
            raise IndexError("pop from empty array")
        return self.delete(self._n - 1)

    def search(self, value):
        """O(n). Returns the first index holding value, or -1."""
        for i in range(self._n):
            if self._store[i] == value:
                return i
        return -1

    def to_list(self):
        return [self._store[i] for i in range(self._n)]

    def __repr__(self):
        return f"DynamicArray({self.to_list()}, cap={self._capacity})"


class Matrix:
    """
    A 2-D matrix stored row-major in one flat DynamicArray.

    Element (r, c) lives at flat index r * cols + c, so access and
    assignment are O(1): one multiply and one add, no pointer chasing.

    Operation             Time
    --------------------- --------------------
    get(r, c) / set(r, c) O(1)
    insert_row            O(rows * cols)
    delete_row            O(rows * cols)
    transpose             O(rows * cols)
    row_sum               O(cols)
    column_sum            O(rows)

    Row-major layout is why row_sum is faster than column_sum in practice
    even though both are linear in their own dimension: a row is contiguous
    in memory and a column strides across it.
    """

    def __init__(self, rows, cols, fill=0):
        if rows < 0 or cols < 0:
            raise ValueError("dimensions must be non-negative")
        self.rows = rows
        self.cols = cols
        self._data = DynamicArray(max(1, rows * cols))
        for _ in range(rows * cols):
            self._data.append(fill)

    def _flat(self, r, c):
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise IndexError(f"({r}, {c}) out of range "
                             f"for {self.rows}x{self.cols}")
        return r * self.cols + c

    def get(self, r, c):
        """O(1)."""
        return self._data.access(self._flat(r, c))

    def set(self, r, c, value):
        """O(1)."""
        self._data.set(self._flat(r, c), value)

    def insert_row(self, r, values):
        """O(rows*cols): every element below row r shifts by one full row."""
        if len(values) != self.cols:
            raise ValueError(f"row needs {self.cols} values")
        if not 0 <= r <= self.rows:
            raise IndexError("row index out of range")
        base = r * self.cols
        for offset, v in enumerate(values):
            self._data.insert(base + offset, v)
        self.rows += 1

    def delete_row(self, r):
        """O(rows*cols)."""
        if not 0 <= r < self.rows:
            raise IndexError("row index out of range")
        base = r * self.cols
        removed = [self._data.delete(base) for _ in range(self.cols)]
        self.rows -= 1
        return removed

    def row_sum(self, r):
        """O(cols), walking contiguous memory."""
        return sum(self.get(r, c) for c in range(self.cols))

    def column_sum(self, c):
        """O(rows), striding by self.cols each step."""
        return sum(self.get(r, c) for r in range(self.rows))

    def transpose(self):
        """O(rows*cols). Returns a new Matrix."""
        t = Matrix(self.cols, self.rows)
        for r in range(self.rows):
            for c in range(self.cols):
                t.set(c, r, self.get(r, c))
        return t

    def to_lists(self):
        return [[self.get(r, c) for c in range(self.cols)]
                for r in range(self.rows)]


# ---------------------------------------------------------------------------
# Stacks and queues over arrays
# ---------------------------------------------------------------------------
class ArrayStack:
    """
    LIFO stack on a DynamicArray.

    push O(1) amortized, pop O(1) amortized, peek O(1), is_empty O(1).

    A stack is the one structure where an array has no downside: both ends
    of the work happen at the high index, which is exactly where an array
    is cheap. Nothing ever shifts.
    """

    def __init__(self):
        self._a = DynamicArray()

    def __len__(self):
        return len(self._a)

    def is_empty(self):
        return len(self._a) == 0

    def push(self, value):
        self._a.append(value)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._a.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek at empty stack")
        return self._a.access(len(self._a) - 1)


class ArrayQueueNaive:
    """
    FIFO queue that keeps the front at index 0.

    enqueue O(1) amortized, dequeue O(n).

    Included as a deliberate counter-example. Holding the front at index 0
    means every dequeue closes the hole by shifting all n-1 survivors left,
    so n enqueues followed by n dequeues costs Theta(n^2) overall.
    benchmarks.py measures this against the circular version.
    """

    def __init__(self):
        self._a = DynamicArray()

    def __len__(self):
        return len(self._a)

    def is_empty(self):
        return len(self._a) == 0

    def enqueue(self, value):
        self._a.append(value)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._a.delete(0)       # the O(n) step

    def peek(self):
        if self.is_empty():
            raise IndexError("peek at empty queue")
        return self._a.access(0)


class ArrayQueueCircular:
    """
    FIFO queue on a circular buffer.

    enqueue O(1) amortized, dequeue O(1), peek O(1).

    The fix is to stop insisting the front sits at index 0. A head index
    and a count are enough; both ends move forward modulo the capacity, so
    neither operation ever shifts an element. The cost is that the buffer
    has to be unrolled into order when it grows, which is the only O(n)
    step and happens on a doubling schedule.
    """

    def __init__(self, capacity=8):
        self._capacity = max(1, capacity)
        self._store = [None] * self._capacity
        self._head = 0
        self._n = 0
        self.resizes = 0

    def __len__(self):
        return self._n

    def is_empty(self):
        return self._n == 0

    def _grow(self):
        """Unroll the wrapped contents into a larger block, in order. O(n)."""
        new_store = [None] * (self._capacity * 2)
        for i in range(self._n):
            new_store[i] = self._store[(self._head + i) % self._capacity]
        self._store = new_store
        self._capacity *= 2
        self._head = 0
        self.resizes += 1

    def enqueue(self, value):
        if self._n == self._capacity:
            self._grow()
        tail = (self._head + self._n) % self._capacity
        self._store[tail] = value
        self._n += 1

    def dequeue(self):
        if self._n == 0:
            raise IndexError("dequeue from empty queue")
        value = self._store[self._head]
        self._store[self._head] = None
        self._head = (self._head + 1) % self._capacity
        self._n -= 1
        return value

    def peek(self):
        if self._n == 0:
            raise IndexError("peek at empty queue")
        return self._store[self._head]

    def to_list(self):
        return [self._store[(self._head + i) % self._capacity]
                for i in range(self._n)]


# ---------------------------------------------------------------------------
# Singly linked list
# ---------------------------------------------------------------------------
class Node:
    """One link. __slots__ keeps the per-node overhead down."""
    __slots__ = ("value", "next")

    def __init__(self, value, nxt=None):
        self.value = value
        self.next = nxt


class SinglyLinkedList:
    """
    Singly linked list with head and tail pointers and a maintained count.

    Operation                 Time    Why
    ------------------------- ------- ------------------------------------
    insert_front(v)           O(1)    relink head
    insert_back(v)            O(1)    tail pointer; O(n) without one
    insert_at(i, v)           O(n)    must walk to position i
    delete_front()            O(1)    relink head
    delete_back()             O(n)    singly linked: no way back to the
                                      new tail, so the list is rewalked
    delete_value(v)           O(n)    search then relink
    access(i)                 O(n)    no address arithmetic available
    search(v)                 O(n)
    traverse()                O(n)
    reverse()                 O(1) space, O(n) time

    delete_back being O(n) while delete_front is O(1) is the asymmetry that
    decides which end a linked queue should use for which operation.
    """

    def __init__(self):
        self.head = None
        self.tail = None
        self._n = 0

    def __len__(self):
        return self._n

    def is_empty(self):
        return self._n == 0

    def insert_front(self, value):
        """O(1)."""
        node = Node(value, self.head)
        self.head = node
        if self.tail is None:
            self.tail = node
        self._n += 1

    def insert_back(self, value):
        """O(1) thanks to the tail pointer."""
        node = Node(value)
        if self.tail is None:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        self._n += 1

    def insert_at(self, i, value):
        """O(n): walk i steps, then relink."""
        if not 0 <= i <= self._n:
            raise IndexError(f"index {i} out of range for insert")
        if i == 0:
            return self.insert_front(value)
        if i == self._n:
            return self.insert_back(value)
        prev = self.head
        for _ in range(i - 1):
            prev = prev.next
        prev.next = Node(value, prev.next)
        self._n += 1

    def delete_front(self):
        """O(1)."""
        if self.head is None:
            raise IndexError("delete from empty list")
        value = self.head.value
        self.head = self.head.next
        if self.head is None:
            self.tail = None
        self._n -= 1
        return value

    def delete_back(self):
        """
        O(n). A singly linked node cannot reach its predecessor, so finding
        the new tail means walking the list again. This is the operation a
        doubly linked list exists to fix.
        """
        if self.head is None:
            raise IndexError("delete from empty list")
        if self.head is self.tail:
            value = self.head.value
            self.head = self.tail = None
            self._n -= 1
            return value
        prev = self.head
        while prev.next is not self.tail:
            prev = prev.next
        value = self.tail.value
        prev.next = None
        self.tail = prev
        self._n -= 1
        return value

    def delete_value(self, value):
        """O(n). Removes the first node holding value; True if removed."""
        prev = None
        cur = self.head
        while cur is not None:
            if cur.value == value:
                if prev is None:
                    self.head = cur.next
                else:
                    prev.next = cur.next
                if cur is self.tail:
                    self.tail = prev
                self._n -= 1
                return True
            prev, cur = cur, cur.next
        return False

    def access(self, i):
        """O(n). The cost an array avoids entirely."""
        if not 0 <= i < self._n:
            raise IndexError(f"index {i} out of range for length {self._n}")
        cur = self.head
        for _ in range(i):
            cur = cur.next
        return cur.value

    def search(self, value):
        """O(n). Returns the index of the first match, or -1."""
        cur = self.head
        i = 0
        while cur is not None:
            if cur.value == value:
                return i
            cur = cur.next
            i += 1
        return -1

    def traverse(self):
        """O(n). Yields each value in order."""
        cur = self.head
        while cur is not None:
            yield cur.value
            cur = cur.next

    def reverse(self):
        """O(n) time, O(1) extra space: flip each next pointer in one pass."""
        prev = None
        cur = self.head
        self.tail = self.head
        while cur is not None:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        self.head = prev

    def to_list(self):
        return list(self.traverse())

    def __repr__(self):
        return f"SinglyLinkedList({self.to_list()})"


class LinkedStack:
    """
    Stack on a singly linked list, working entirely at the head.

    push O(1), pop O(1), peek O(1) -- all worst case, not amortized, since
    nothing is ever resized or copied.
    """

    def __init__(self):
        self._list = SinglyLinkedList()

    def __len__(self):
        return len(self._list)

    def is_empty(self):
        return self._list.is_empty()

    def push(self, value):
        self._list.insert_front(value)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._list.delete_front()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek at empty stack")
        return self._list.head.value


class LinkedQueue:
    """
    Queue on a singly linked list: enqueue at the tail, dequeue at the head.

    enqueue O(1), dequeue O(1), peek O(1), all worst case.

    Picking the ends this way matters. Head-dequeue and tail-enqueue are
    both O(1); the reverse assignment would make dequeue O(n), because a
    singly linked list cannot step backwards from its tail.
    """

    def __init__(self):
        self._list = SinglyLinkedList()

    def __len__(self):
        return len(self._list)

    def is_empty(self):
        return self._list.is_empty()

    def enqueue(self, value):
        self._list.insert_back(value)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._list.delete_front()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek at empty queue")
        return self._list.head.value


# ---------------------------------------------------------------------------
# Rooted tree (optional part of the assignment)
# ---------------------------------------------------------------------------
class TreeNode:
    """
    A rooted-tree node in the left-child / right-sibling representation.

    A node with an unbounded number of children stores only two pointers:
    its first child, and its next sibling. The children of a node are
    therefore a singly linked list hanging off first_child. This keeps the
    space per node at O(1) no matter how wide the tree gets, which the
    alternative -- an array of child pointers per node -- does not.
    """
    __slots__ = ("value", "first_child", "next_sibling", "parent")

    def __init__(self, value):
        self.value = value
        self.first_child = None
        self.next_sibling = None
        self.parent = None


class RootedTree:
    """
    Rooted tree built from linked TreeNodes.

    Operation            Time      Why
    -------------------- --------- --------------------------------------
    add_child(p, v)      O(1)      prepend to p's child list
    children(node)       O(c)      c = that node's child count
    traverse_preorder    O(n)      every node and edge once
    traverse_bfs         O(n)      uses the LinkedQueue above
    search(v)            O(n)      no ordering to exploit
    height()             O(n)
    size()               O(1)      maintained on insert

    add_child prepends rather than appends, which is what makes it O(1):
    appending would require walking the sibling list to its end.
    """

    def __init__(self, root_value=None):
        self.root = TreeNode(root_value) if root_value is not None else None
        self._n = 1 if self.root else 0

    def size(self):
        return self._n

    def add_child(self, parent, value):
        """O(1). Returns the new node."""
        if parent is None:
            raise ValueError("parent must not be None")
        node = TreeNode(value)
        node.parent = parent
        node.next_sibling = parent.first_child
        parent.first_child = node
        self._n += 1
        return node

    def children(self, node):
        """O(c) for a node with c children."""
        cur = node.first_child
        out = []
        while cur is not None:
            out.append(cur)
            cur = cur.next_sibling
        return out

    def traverse_preorder(self, node=None):
        """
        O(n). Explicit stack rather than recursion, so a deep or degenerate
        tree cannot overflow the interpreter's call stack.
        """
        start = node if node is not None else self.root
        if start is None:
            return
        stack = LinkedStack()
        stack.push(start)
        while not stack.is_empty():
            cur = stack.pop()
            yield cur.value
            # Push siblings in reverse so the first child comes out first.
            kids = self.children(cur)
            for child in reversed(kids):
                stack.push(child)

    def traverse_bfs(self, node=None):
        """O(n). Level order, using the LinkedQueue from this module."""
        start = node if node is not None else self.root
        if start is None:
            return
        q = LinkedQueue()
        q.enqueue(start)
        while not q.is_empty():
            cur = q.dequeue()
            yield cur.value
            for child in self.children(cur):
                q.enqueue(child)

    def search(self, value):
        """O(n). Returns the node or None."""
        if self.root is None:
            return None
        stack = LinkedStack()
        stack.push(self.root)
        while not stack.is_empty():
            cur = stack.pop()
            if cur.value == value:
                return cur
            for child in self.children(cur):
                stack.push(child)
        return None

    def height(self):
        """O(n). Edge count on the longest root-to-leaf path."""
        if self.root is None:
            return -1
        best = 0
        stack = LinkedStack()
        stack.push((self.root, 0))
        while not stack.is_empty():
            node, d = stack.pop()
            if d > best:
                best = d
            for child in self.children(node):
                stack.push((child, d + 1))
        return best

    def depth_of(self, node):
        """O(h). Walks parent pointers up to the root."""
        d = 0
        cur = node
        while cur.parent is not None:
            cur = cur.parent
            d += 1
        return d


# ---------------------------------------------------------------------------
# Correctness suite
# ---------------------------------------------------------------------------
def run_correctness_suite():
    import random
    rng = random.Random(999)
    results = []

    def check(label, condition):
        results.append((label, bool(condition)))

    # --- DynamicArray: mirror every operation against a reference list ---
    da = DynamicArray(capacity=2)
    ref = []
    for _ in range(600):
        op = rng.random()
        if op < 0.45 or not ref:
            v = rng.randint(0, 999)
            da.append(v)
            ref.append(v)
        elif op < 0.65:
            i = rng.randrange(len(ref) + 1)
            v = rng.randint(0, 999)
            da.insert(i, v)
            ref.insert(i, v)
        elif op < 0.85:
            i = rng.randrange(len(ref))
            check("DynamicArray.delete returns element",
                  da.delete(i) == ref.pop(i))
        else:
            i = rng.randrange(len(ref))
            check("DynamicArray.access", da.access(i) == ref[i])
    check("DynamicArray matches reference after 600 ops",
          da.to_list() == ref)
    check("DynamicArray length tracked", len(da) == len(ref))
    check("DynamicArray capacity >= length", da.capacity >= len(da))
    check("DynamicArray search finds element",
          ref and da.search(ref[len(ref) // 2]) != -1)
    check("DynamicArray search misses absent element", da.search(-1) == -1)

    shrink = DynamicArray(capacity=8)
    for i in range(200):
        shrink.append(i)
    big_cap = shrink.capacity
    for _ in range(195):
        shrink.pop()
    check("DynamicArray shrinks when sparse", shrink.capacity < big_cap)

    # --- Matrix ---
    m = Matrix(3, 4)
    for r in range(3):
        for c in range(4):
            m.set(r, c, r * 10 + c)
    check("Matrix get/set", m.get(2, 3) == 23)
    check("Matrix row_sum", m.row_sum(1) == 10 + 11 + 12 + 13)
    check("Matrix column_sum", m.column_sum(2) == 2 + 12 + 22)
    t = m.transpose()
    check("Matrix transpose shape", (t.rows, t.cols) == (4, 3))
    check("Matrix transpose values", t.get(3, 2) == m.get(2, 3))
    m.insert_row(1, [100, 101, 102, 103])
    check("Matrix insert_row shifts", m.rows == 4 and m.get(1, 2) == 102)
    check("Matrix insert_row preserves below", m.get(2, 2) == 12)
    removed = m.delete_row(1)
    check("Matrix delete_row returns row", removed == [100, 101, 102, 103])
    check("Matrix delete_row restores", m.rows == 3 and m.get(1, 2) == 12)

    # --- Stacks: both implementations must behave identically ---
    for name, stack in (("ArrayStack", ArrayStack()),
                        ("LinkedStack", LinkedStack())):
        ref_stack = []
        for _ in range(400):
            if rng.random() < 0.6 or not ref_stack:
                v = rng.randint(0, 999)
                stack.push(v)
                ref_stack.append(v)
            else:
                check(f"{name} pop order", stack.pop() == ref_stack.pop())
        check(f"{name} length", len(stack) == len(ref_stack))
        check(f"{name} peek", (not ref_stack) or stack.peek() == ref_stack[-1])

    # --- Queues: all three implementations must agree ---
    from collections import deque
    for name, queue in (("ArrayQueueNaive", ArrayQueueNaive()),
                        ("ArrayQueueCircular", ArrayQueueCircular(capacity=2)),
                        ("LinkedQueue", LinkedQueue())):
        ref_q = deque()
        for _ in range(400):
            if rng.random() < 0.6 or not ref_q:
                v = rng.randint(0, 999)
                queue.enqueue(v)
                ref_q.append(v)
            else:
                check(f"{name} FIFO order", queue.dequeue() == ref_q.popleft())
        check(f"{name} length", len(queue) == len(ref_q))
        check(f"{name} peek", (not ref_q) or queue.peek() == ref_q[0])

    # Circular buffer must survive wrapping past the end of its block.
    cq = ArrayQueueCircular(capacity=4)
    for i in range(4):
        cq.enqueue(i)
    for _ in range(3):
        cq.dequeue()
    for i in range(10, 14):
        cq.enqueue(i)
    check("Circular queue wraps correctly",
          cq.to_list() == [3, 10, 11, 12, 13])

    # --- SinglyLinkedList ---
    ll = SinglyLinkedList()
    for v in (3, 2, 1):
        ll.insert_front(v)
    for v in (4, 5):
        ll.insert_back(v)
    check("LinkedList order", ll.to_list() == [1, 2, 3, 4, 5])
    ll.insert_at(2, 99)
    check("LinkedList insert_at", ll.to_list() == [1, 2, 99, 3, 4, 5])
    check("LinkedList access", ll.access(2) == 99)
    check("LinkedList search hit", ll.search(99) == 2)
    check("LinkedList search miss", ll.search(1000) == -1)
    check("LinkedList delete_value", ll.delete_value(99) is True)
    check("LinkedList after delete_value", ll.to_list() == [1, 2, 3, 4, 5])
    check("LinkedList delete_front", ll.delete_front() == 1)
    check("LinkedList delete_back", ll.delete_back() == 5)
    check("LinkedList after end deletes", ll.to_list() == [2, 3, 4])
    check("LinkedList length", len(ll) == 3)
    ll.reverse()
    check("LinkedList reverse", ll.to_list() == [4, 3, 2])
    check("LinkedList tail after reverse", ll.tail.value == 2)
    ll.insert_back(1)
    check("LinkedList tail usable after reverse",
          ll.to_list() == [4, 3, 2, 1])

    single = SinglyLinkedList()
    single.insert_front(7)
    check("LinkedList single delete_back", single.delete_back() == 7)
    check("LinkedList empty after", single.is_empty() and single.tail is None)

    # --- RootedTree ---
    tree = RootedTree("root")
    a = tree.add_child(tree.root, "a")
    b = tree.add_child(tree.root, "b")
    tree.add_child(a, "a1")
    tree.add_child(a, "a2")
    deep = tree.add_child(b, "b1")
    for i in range(2, 6):
        deep = tree.add_child(deep, f"b{i}")
    check("Tree size", tree.size() == 10)
    # root -> b -> b1 -> b2 -> b3 -> b4 -> b5 is 6 edges deep
    check("Tree height", tree.height() == 6)
    check("Tree depth_of", tree.depth_of(deep) == 6)
    pre = list(tree.traverse_preorder())
    bfs = list(tree.traverse_bfs())
    check("Tree preorder visits all", len(pre) == 10)
    check("Tree preorder starts at root", pre[0] == "root")
    check("Tree bfs visits all", len(bfs) == 10)
    check("Tree bfs level order", bfs[:3] == ["root", "b", "a"])
    check("Tree search hit", tree.search("a2") is not None)
    check("Tree search miss", tree.search("nope") is None)

    # --- Error handling ---
    errors = 0
    for fn in (lambda: DynamicArray().access(0),
               lambda: DynamicArray().pop(),
               lambda: ArrayStack().pop(),
               lambda: LinkedStack().pop(),
               lambda: ArrayQueueCircular().dequeue(),
               lambda: LinkedQueue().dequeue(),
               lambda: SinglyLinkedList().delete_front(),
               lambda: Matrix(2, 2).get(5, 0)):
        try:
            fn()
        except (IndexError, ValueError):
            errors += 1
    check("Empty/out-of-range operations raise", errors == 8)

    failures = [label for label, ok in results if not ok]
    print(f"Data structures: {len(results) - len(failures)}/{len(results)} "
          f"checks passed")
    for label in failures:
        print(f"  FAIL  {label}")
    return not failures


if __name__ == "__main__":
    ok = run_correctness_suite()
    raise SystemExit(0 if ok else 1)
