# Assignment 6 — Medians and Order Statistics & Elementary Data Structures

MSCS-532 Algorithms and Data Structures
Fortunato Hernandez

Part 1 implements selection of the k-th smallest element two ways
(deterministic Median of Medians, randomized Quickselect) and measures them.
Part 2 implements arrays, matrices, stacks, queues, singly linked lists, and
rooted trees from scratch and measures their operations.

## Files

| File | What it holds |
|---|---|
| `selection.py` | Median of Medians (groups of 3, 5, 7), randomized Quickselect, and a Lomuto-partition variant kept as a counter-example. Run it to execute the correctness suite. |
| `data_structures.py` | `DynamicArray`, `Matrix`, `ArrayStack`, `ArrayQueueNaive`, `ArrayQueueCircular`, `SinglyLinkedList`, `LinkedStack`, `LinkedQueue`, `RootedTree`. Run it for its correctness suite. |
| `benchmarks.py` | All measurements. Writes seven CSV files. |
| `make_charts.py` | Builds the seven figures from those CSV files. |
| `Assignment6_Report.docx` | The report. |

## How to run

Python 3.9 or newer. Only `matplotlib` is needed, and only for the figures.

```bash
pip install matplotlib

python selection.py        # 2,910 correctness checks on the selection algorithms
python data_structures.py  # 1,054 correctness checks on the data structures
python benchmarks.py       # all measurements -> 7 CSV files (about 2.5 minutes)
python make_charts.py      # 7 figures from those CSV files
```

`selection.py` and `data_structures.py` exit non-zero if any check fails, so
they work as a test suite. Both verify against an independent reference:
every selection result is checked against `sorted(data)[k]` for *every* valid
k, and every data structure is mirrored against a Python list or `deque`
across several hundred randomized operations.

## Findings

**Part 1 — selection**

1. **Both algorithms are linear, and the guarantee costs about 2x.** On
   100,000 random integers, Median of Medians (groups of 5) used 10.11
   comparisons per element against Quickselect's 5.46, a ratio of 1.85 (and
   2.2x to 2.7x in wall-clock time) for the same answer. Asymptotic equality
   says nothing about which one to run.

2. **The group size is not a tuning knob; it sets the complexity class.**
   Over a 256x range in n, groups of 5 flattened out near 10.3 comparisons
   per element while groups of 3 climbed steadily from 11.94 to 14.14 and
   showed no sign of settling. Dividing instead by `n log n` reverses which
   curve is flat, which is the signature of `n log n` growth.

3. **The reason is the pivot search, not the partition.** Groups of 3 are
   actually the *better* partitioner — their guarantee is 2n/3 surviving
   against 7n/10 for groups of 5. What breaks them is that finding the pivot
   costs `T(n/3)` instead of `T(n/5)`. Once the two fractions sum to exactly
   1 instead of 9/10, the work stops shrinking from level to level. Splitting
   the comparison count by where it was spent confirms it: the partitioning
   share is what grows.

4. **Stack depth follows log_g(n) exactly.** Because the surviving-side
   recursion is written as a loop, the only thing on the stack is the pivot
   search. At n = 2,048,000 that measured 13 frames for groups of 3, 9 for
   groups of 5, and 7 for groups of 7 — against log_3 = 13.2, log_5 = 9.0,
   log_7 = 7.5. Quickselect needs 0 extra frames.

5. **Duplicates break the partition scheme, and no pivot rule can save it.**
   With a plain Lomuto partition, an array of identical keys cost exactly
   `n(n-1)/2 - k(k-1)/2` comparisons — matched to the digit at four
   different sizes, with the ratio to n² holding at 0.375 = 3/8. A random
   pivot does not help, because when every key is equal every pivot is the
   same pivot. The three-way partition finished the same input in exactly
   2n comparisons, because one pass puts every equal key in the middle block
   and the answer is already there. At n = 5,000 that is a 1,924x difference
   in wall-clock time. The three-way partition pays for this with about 2x
   the comparisons on fully distinct keys — that is the premium for the
   insurance.

6. **"Expected linear" is an average, and the spread is wide.** Running
   Quickselect 500 times on one fixed 50,000-element array, comparisons per
   element ranged from 2.16 to 12.31 — a 5.7x spread, with the worst seed
   costing more than Median of Medians always costs. Median of Medians never
   let more than 52.7% of the array survive a round, inside its proven 70%
   ceiling. Quickselect had a round leaving more than 70% standing in 487 of
   500 seeds, and more than 90% in 405 of 500.

7. **Structured input can make the deterministic algorithm finish in one
   round.** On sorted, reverse-sorted, and duplicate-heavy input, the
   constructed pivot sometimes *is* the element being sought, and the
   algorithm returns immediately. That happened in 14 of 36 structured-input
   cells and in 0 of 12 random-input cells. It depends on how n divides into
   groups, so it lands unpredictably — which is what makes the timings on
   structured input jump around rather than curve smoothly.

**Part 2 — data structures**

8. **Where the complexity classes differ, the gap is enormous.** At n =
   16,000: indexed access was 2,322x faster on the array; inserting at the
   front was 1,363x faster on the linked list; and a queue holding its front
   at index 0 was 919x slower than the same queue on a circular buffer.
   These are the differences worth designing around.

9. **Where the complexity classes match, almost nothing is left.** Array and
   linked-list sequential traversal are both O(n), and measured like for
   like they stayed within about 20% of each other, with the linked list
   slightly ahead. The array's textbook cache-locality advantage did not
   appear. The row-major matrix scan did beat the column-major scan, but
   only by 4-7%, and the direction of that gap flipped between runs, so it
   sits inside run-to-run variation rather than a real effect. The honest
   reason is the language: in CPython a list holds *pointers* to
   separately allocated integer objects, so neither traversal order walks
   the values contiguously, and interpreter overhead per element dwarfs
   the memory-access cost the effect would show up in. This is a limit of
   measuring locality in Python, not evidence that locality does not
   matter.

10. **Measuring through a public API can invert the result.** The same
    traversal comparison run through `access(i)` — a method call plus a
    bounds check per element — made the linked list look 2.3x faster than
    the array, instead of the 1.2x the fair comparison shows. That gap was
    the wrapper, not the structure. Both versions are in
    `ds_operations.csv`.

11. **Fixing the array queue costs nothing asymptotically.** The problem is
    insisting the front sits at index 0, which forces every dequeue to shift
    n-1 elements. A head index plus a count removes the shifting entirely
    and makes both ends O(1), leaving the growth copy as the only O(n) step.
    Doubling kept that cheap: 100,000 appends took 14 resizes.

## Operation complexities

| Structure | Access | Search | Insert | Delete |
|---|---|---|---|---|
| `DynamicArray` | O(1) | O(n) | O(1) amortized at the end, O(n) at an index | O(1) amortized at the end, O(n) at an index |
| `Matrix` (row-major) | O(1) | O(rows·cols) | O(rows·cols) per row | O(rows·cols) per row |
| `ArrayStack` | O(1) peek | O(n) | O(1) amortized push | O(1) amortized pop |
| `ArrayQueueCircular` | O(1) peek | O(n) | O(1) amortized enqueue | O(1) dequeue |
| `ArrayQueueNaive` | O(1) peek | O(n) | O(1) amortized enqueue | **O(n)** dequeue |
| `SinglyLinkedList` | O(n) | O(n) | O(1) at either end, O(n) at an index | O(1) at the front, O(n) at the back or by value |
| `LinkedStack` | O(1) peek | O(n) | O(1) push | O(1) pop |
| `LinkedQueue` | O(1) peek | O(n) | O(1) enqueue | O(1) dequeue |
| `RootedTree` | O(1) add child | O(n) | O(1) | — |

Selection, for arrays of n elements:

| Algorithm | Time | Extra space | Guarantee |
|---|---|---|---|
| Median of Medians (g=5) | Θ(n) worst case | O(log n) stack, in place otherwise | every round eliminates ≥ 30% |
| Randomized Quickselect | O(n) expected, O(n²) worst case | O(1), in place | none per round |

## Output files

`benchmarks.py` writes `sel_timing.csv`, `sel_early_termination.csv`,
`sel_constants.csv`, `sel_groupsize.csv`, `sel_duplicates.csv`,
`sel_variance.csv`, and `ds_operations.csv`. `make_charts.py` reads those and
writes `fig1`–`fig7` as PNG files, so the figures and the report tables come
from the same measurements.
