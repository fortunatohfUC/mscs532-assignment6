"""
MSCS-532 Assignment 6, Part 1: Medians and Order Statistics

Selection of the k-th smallest element, three ways:

  1. deterministic_select  - Median of Medians, O(n) worst case
  2. randomized_select     - Randomized Quickselect with a three-way
                             partition, O(n) expected
  3. randomized_select_lomuto - Randomized Quickselect with the plain
                             Lomuto partition. Included on purpose as a
                             counter-example: it is O(n) expected on
                             distinct keys but degrades to O(n^2) when the
                             array is full of duplicates.

Run:  python selection.py          (runs the correctness suite)

Implementation notes
--------------------
Every algorithm here partitions the array in place and then *loops* on the
surviving side instead of calling itself again. The recursive call is in
tail position, so turning it into a loop is a safe rewrite and it keeps the
call stack from growing with n. For Quickselect that leaves O(1) extra
space. For Median of Medians the stack still grows, but only through the
pivot-finding call on n/g elements, which is O(log n) frames, not O(n).

Counting is done through a Counters object that is passed down the call
chain. Comparisons and swaps are counted at the point where they happen,
which makes the numbers exact rather than sampled.
"""

import random
from dataclasses import dataclass


@dataclass
class Counters:
    """
    Instrumentation for one call. Exact counts, not estimates.

    The comparison total is also split by where the work happened, because
    that split is what explains the difference between group sizes in
    Median of Medians:

      group_comparisons     - spent sorting groups to find a pivot, i.e.
                              the price of the pivot-finding recursion
      partition_comparisons - spent in the partition passes themselves

    rounds and worst_shrink describe the outermost elimination loop only:
    how many partition rounds the top-level range needed, and the largest
    fraction of it that ever survived a single round. worst_shrink is the
    quantity the theoretical guarantees bound (<= 7/10 for groups of 5,
    <= 2/3 for groups of 3). partitions, by contrast, counts every
    partition pass anywhere in the computation, nested pivot-finding
    included.
    """
    comparisons: int = 0
    group_comparisons: int = 0
    partition_comparisons: int = 0
    swaps: int = 0
    partitions: int = 0      # number of partition passes performed
    rounds: int = 0          # main-loop iterations (elimination rounds)
    max_stack: int = 0       # deepest nesting of pivot-finding recursion
    worst_shrink: float = 0.0
    worst_shrink_large: float = 0.0

    # The 7/10 guarantee is asymptotic. On a range of 8 elements, ceilings
    # mean 6 can survive, which is 0.75 and looks like a broken bound. So
    # the shrink ratio is recorded twice: over every round, and over only
    # those rounds whose incoming range was at least this big, which is the
    # measurement that can fairly be held against the asymptotic claim.
    LARGE_RANGE = 1_000

    def note_depth(self, d):
        if d > self.max_stack:
            self.max_stack = d

    def note_shrink(self, fraction, range_size):
        # range_size is required on purpose. An optional default here let a
        # caller silently skip the large-range tracking and report 0.0,
        # which looked like a real measurement.
        if fraction > self.worst_shrink:
            self.worst_shrink = fraction
        if (range_size >= self.LARGE_RANGE
                and fraction > self.worst_shrink_large):
            self.worst_shrink_large = fraction


# ---------------------------------------------------------------------------
# Shared partition routines
# ---------------------------------------------------------------------------
def _three_way_partition(arr, low, high, pivot_val, ctr):
    """
    Dutch-national-flag partition around a pivot *value*.

    Rearranges arr[low..high] into three blocks and returns (lt, gt) with

        arr[low .. lt-1]  <  pivot_val
        arr[lt  .. gt  ]  == pivot_val
        arr[gt+1.. high]  >  pivot_val

    The middle block is why this version survives duplicate-heavy input:
    every element equal to the pivot is removed from the problem in this
    one pass, so the next round cannot be handed back the same elements.
    """
    ctr.partitions += 1
    lt, i, gt = low, low, high

    while i <= gt:
        ctr.comparisons += 1
        ctr.partition_comparisons += 1
        if arr[i] < pivot_val:
            arr[lt], arr[i] = arr[i], arr[lt]
            ctr.swaps += 1
            lt += 1
            i += 1
        else:
            ctr.comparisons += 1
            ctr.partition_comparisons += 1
            if arr[i] > pivot_val:
                arr[i], arr[gt] = arr[gt], arr[i]
                ctr.swaps += 1
                gt -= 1
            else:
                i += 1

    return lt, gt


def _lomuto_partition(arr, low, high, ctr):
    """
    Classic Lomuto partition around arr[high]; returns the pivot's final
    index. Elements equal to the pivot all land on the low side, which is
    exactly the behaviour that makes it collapse on duplicate keys.
    """
    ctr.partitions += 1
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        ctr.comparisons += 1
        ctr.partition_comparisons += 1
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            ctr.swaps += 1

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    ctr.swaps += 1
    return i + 1


def _insertion_sort(arr, low, high, ctr):
    """
    Sort the small range arr[low..high] in place. Used only on groups of
    g elements inside Median of Medians, where g is a constant (3 or 5),
    so each call is O(1) work.
    """
    for i in range(low + 1, high + 1):
        key = arr[i]
        j = i - 1
        while j >= low:
            ctr.comparisons += 1
            ctr.group_comparisons += 1
            if arr[j] > key:
                arr[j + 1] = arr[j]
                ctr.swaps += 1
                j -= 1
            else:
                break
        arr[j + 1] = key


# ---------------------------------------------------------------------------
# 1. Deterministic selection: Median of Medians
# ---------------------------------------------------------------------------
def _median_of_medians(arr, low, high, g, ctr, depth):
    """
    Produce a pivot value guaranteed to be near the middle of
    arr[low..high].

    Split the range into ceil(n/g) groups of g, sort each group, and move
    each group's median into the front of the range. Then select the median
    of that block of medians recursively.

    Moving medians to the front is safe because groups are processed left to
    right: the slot low+i that group i's median is swapped into always sits
    inside a group that has already given up its median, so nothing
    unprocessed is disturbed and no element is lost.
    """
    n = high - low + 1
    num_groups = (n + g - 1) // g

    for i in range(num_groups):
        gl = low + i * g
        gh = min(gl + g - 1, high)
        _insertion_sort(arr, gl, gh, ctr)
        median_idx = gl + (gh - gl) // 2
        arr[low + i], arr[median_idx] = arr[median_idx], arr[low + i]
        ctr.swaps += 1

    # The medians now occupy arr[low .. low+num_groups-1].
    # Take their median: 0-indexed rank num_groups//2.
    return _select_mom(arr, low, low + num_groups - 1,
                       num_groups // 2, g, ctr, depth)


def _select_mom(arr, low, high, k, g, ctr, depth=0):
    """
    Return the (k+1)-th smallest value of arr[low..high], k 0-indexed.
    Loops on the surviving side; recurses only to find the pivot.
    """
    ctr.note_depth(depth)

    while True:
        n = high - low + 1

        # Base case: a range no bigger than one group is sorted directly.
        if n <= g:
            _insertion_sort(arr, low, high, ctr)
            return arr[low + k]

        if depth == 0:
            ctr.rounds += 1
        pivot_val = _median_of_medians(arr, low, high, g, ctr, depth + 1)
        lt, gt = _three_way_partition(arr, low, high, pivot_val, ctr)

        left_count = lt - low
        mid_count = gt - lt + 1

        if k < left_count:
            high = lt - 1
        elif k < left_count + mid_count:
            return pivot_val          # the pivot itself is the answer
        else:
            k -= left_count + mid_count
            low = gt + 1

        # Only the outermost call's shrinkage is comparable to the 7/10
        # guarantee; the nested pivot-finding calls run on their own tiny
        # ranges where the ratios are coarse and mean something different.
        if depth == 0:
            ctr.note_shrink((high - low + 1) / n, n)


def deterministic_select(arr, k, group_size=5, counters=None, in_place=False):
    """
    Median of Medians selection. Returns the (k+1)-th smallest element,
    k 0-indexed, so k=0 is the minimum and k=len(arr)-1 the maximum.

    group_size is exposed because it is the whole reason the algorithm is
    linear. With g=5 the recurrence is T(n) <= T(n/5) + T(7n/10) + O(n),
    and 1/5 + 7/10 = 9/10 < 1, so the work per level shrinks geometrically.
    With g=3 it becomes T(n) <= T(n/3) + T(2n/3) + O(n), where the fractions
    sum to exactly 1 and the result is Theta(n log n). benchmarks.py
    measures both.

    By default the input list is copied, so callers keep their data. Pass
    in_place=True to skip the copy; the core routine never allocates.
    """
    if not arr:
        raise ValueError("cannot select from an empty array")
    if not 0 <= k < len(arr):
        raise IndexError(f"k={k} out of range for array of length {len(arr)}")
    if group_size < 3 or group_size % 2 == 0:
        raise ValueError("group_size must be an odd integer >= 3")

    ctr = counters if counters is not None else Counters()
    work = arr if in_place else list(arr)
    return _select_mom(work, 0, len(work) - 1, k, group_size, ctr)


# ---------------------------------------------------------------------------
# 2. Randomized Quickselect (three-way partition)
# ---------------------------------------------------------------------------
def randomized_select(arr, k, counters=None, rng=None, in_place=False):
    """
    Randomized Quickselect. Returns the (k+1)-th smallest element.

    The pivot is drawn uniformly at random from the live range, so no fixed
    input ordering is reliably bad. Partitioning is three-way, which means
    runs of equal keys are consumed in a single pass.

    Expected O(n): a uniformly random pivot leaves a range of at most 3n/4
    with probability at least 1/2, so the expected size shrinks by a
    constant factor each round and the total expected work is a geometric
    series summing to O(n). The worst case is still O(n^2), but it depends
    on the random draws rather than on the data.
    """
    if not arr:
        raise ValueError("cannot select from an empty array")
    if not 0 <= k < len(arr):
        raise IndexError(f"k={k} out of range for array of length {len(arr)}")

    ctr = counters if counters is not None else Counters()
    r = rng if rng is not None else random
    work = arr if in_place else list(arr)

    low, high = 0, len(work) - 1

    while True:
        if low == high:
            return work[low]

        n = high - low + 1
        ctr.rounds += 1
        pivot_val = work[r.randint(low, high)]
        lt, gt = _three_way_partition(work, low, high, pivot_val, ctr)

        left_count = lt - low
        mid_count = gt - lt + 1

        if k < left_count:
            high = lt - 1
        elif k < left_count + mid_count:
            return pivot_val
        else:
            k -= left_count + mid_count
            low = gt + 1

        ctr.note_shrink((high - low + 1) / n, n)


# ---------------------------------------------------------------------------
# 3. Randomized Quickselect with Lomuto partition (the counter-example)
# ---------------------------------------------------------------------------
def randomized_select_lomuto(arr, k, counters=None, rng=None, in_place=False):
    """
    Same random pivot, but a two-way Lomuto partition.

    Kept in the submission because it isolates one specific failure. On
    distinct keys it behaves like randomized_select. On an array where many
    keys are equal, Lomuto sends every element equal to the pivot to the low
    side, so the pivot lands at the top of the range and the next round gets
    n-1 elements back. Randomizing the pivot does not help, because when the
    keys are all equal every pivot is the same pivot. The result is
    Theta(n^2).
    """
    if not arr:
        raise ValueError("cannot select from an empty array")
    if not 0 <= k < len(arr):
        raise IndexError(f"k={k} out of range for array of length {len(arr)}")

    ctr = counters if counters is not None else Counters()
    r = rng if rng is not None else random
    work = arr if in_place else list(arr)

    low, high = 0, len(work) - 1

    while True:
        if low == high:
            return work[low]

        n = high - low + 1
        ctr.rounds += 1
        j = r.randint(low, high)
        work[j], work[high] = work[high], work[j]
        ctr.swaps += 1

        p = _lomuto_partition(work, low, high, ctr)

        # k is always expressed relative to `low`, so the pivot's rank
        # inside the live range is p - low.
        if p - low == k:
            return work[p]
        elif k < p - low:
            high = p - 1
        else:
            k -= (p - low) + 1
            low = p + 1

        ctr.note_shrink((high - low + 1) / n, n)


# ---------------------------------------------------------------------------
# Correctness suite
# ---------------------------------------------------------------------------
def _check(name, fn, data, k, expected):
    got = fn(list(data), k)
    status = "ok" if got == expected else "FAIL"
    if status == "FAIL":
        print(f"  {status}  {name}: k={k} got {got}, expected {expected}")
    return status == "ok"


def run_correctness_suite():
    """
    Exhaustive check on small inputs plus the edge cases the assignment
    calls out. Every algorithm is checked against sorted(data)[k] for
    every valid k, not just one spot check.
    """
    rng = random.Random(12345)

    algorithms = {
        "MoM g=5": lambda a, k: deterministic_select(a, k, group_size=5),
        "MoM g=3": lambda a, k: deterministic_select(a, k, group_size=3),
        "MoM g=7": lambda a, k: deterministic_select(a, k, group_size=7),
        "Quickselect (3-way)": lambda a, k: randomized_select(a, k),
        "Quickselect (Lomuto)": lambda a, k: randomized_select_lomuto(a, k),
    }

    cases = []

    # Edge cases named in the assignment and the usual boundary suspects.
    cases.append(("single element", [42]))
    cases.append(("two elements", [9, 4]))
    cases.append(("two equal", [7, 7]))
    cases.append(("all identical", [5] * 40))
    cases.append(("two distinct values", [1, 2] * 20))
    cases.append(("heavy duplicates", [rng.randint(0, 4) for _ in range(60)]))
    cases.append(("already sorted", list(range(50))))
    cases.append(("reverse sorted", list(range(50, 0, -1))))
    cases.append(("negatives and zero", [-5, 0, 3, -5, 12, 0, -1]))
    cases.append(("floats", [0.5, -2.25, 7.0, 0.5, 3.125]))

    # Exhaustive small random arrays: every length 1..25, every k.
    for n in range(1, 26):
        cases.append((f"random n={n}",
                      [rng.randint(0, 30) for _ in range(n)]))

    total = 0
    passed = 0
    for label, data in cases:
        reference = sorted(data)
        for k in range(len(data)):
            for name, fn in algorithms.items():
                total += 1
                if _check(f"{name} / {label}", fn, data, k, reference[k]):
                    passed += 1

    print(f"Correctness: {passed}/{total} checks passed "
          f"across {len(cases)} input cases")

    # Error handling
    errors_ok = 0
    try:
        deterministic_select([], 0)
    except ValueError:
        errors_ok += 1
    try:
        randomized_select([1, 2, 3], 5)
    except IndexError:
        errors_ok += 1
    try:
        deterministic_select([1, 2, 3], 0, group_size=4)
    except ValueError:
        errors_ok += 1
    print(f"Error handling: {errors_ok}/3 invalid inputs rejected")

    # The input array must come back untouched when in_place is False.
    original = [5, 3, 9, 1, 7]
    snapshot = list(original)
    deterministic_select(original, 2)
    randomized_select(original, 2)
    print(f"Caller's array preserved: {original == snapshot}")

    return passed == total


if __name__ == "__main__":
    ok = run_correctness_suite()
    raise SystemExit(0 if ok else 1)
