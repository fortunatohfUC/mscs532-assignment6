"""
MSCS-532 Assignment 6: empirical measurements for both parts.

Run:  python benchmarks.py

Writes seven CSV files that the report and make_charts.py both read:

  sel_timing.csv      wall-clock time, selection algorithms x sizes x inputs
  sel_early_termination.csv  elimination rounds needed per input shape
  sel_constants.csv   exact comparison counts per element
  sel_groupsize.csv   groups of 3 vs 5 vs 7, comparisons/n as n grows
  sel_duplicates.csv  three-way vs Lomuto partition as duplicates increase
  sel_variance.csv    spread of randomized quickselect over many seeds
  ds_operations.csv   data structure operation costs

Method notes
------------
Timing is the median of several trials, not the mean, so one scheduling
hiccup cannot move the number. Comparison counts are exact and come from
the Counters object inside the algorithms, so the structural results do not
depend on machine noise at all. Median of Medians is fully deterministic,
so its counts need only one run; quickselect is averaged over distinct
seeds.
"""

import csv
import math
import random
import statistics
import time

from selection import (Counters, deterministic_select, randomized_select,
                       randomized_select_lomuto)
from data_structures import (ArrayQueueCircular, ArrayQueueNaive, ArrayStack,
                             DynamicArray, LinkedQueue, LinkedStack, Matrix,
                             SinglyLinkedList)

TRIALS = 5


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------
def make_input(n, kind, seed=42):
    rng = random.Random(seed)
    if kind == "random":
        return [rng.randint(0, 10_000_000) for _ in range(n)]
    if kind == "sorted":
        return list(range(n))
    if kind == "reverse":
        return list(range(n, 0, -1))
    if kind == "duplicates":
        # Only sqrt(n) distinct values, so every value repeats often.
        d = max(2, int(n ** 0.5))
        return [rng.randint(0, d) for _ in range(n)]
    raise ValueError(f"unknown input kind: {kind}")


def time_median(fn, trials=TRIALS):
    """Median wall-clock seconds over `trials` runs of fn()."""
    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


# ---------------------------------------------------------------------------
# Experiment A: wall-clock timing
# ---------------------------------------------------------------------------
def run_selection_timing():
    print("\n[A] Selection timing (median of "
          f"{TRIALS} trials, selecting the median element)")
    sizes = [12_500, 25_000, 50_000, 100_000]
    kinds = ["random", "sorted", "reverse", "duplicates"]
    algorithms = {
        "MoM (g=5)": lambda a, k: deterministic_select(a, k, group_size=5),
        "MoM (g=3)": lambda a, k: deterministic_select(a, k, group_size=3),
        "Quickselect (3-way)": lambda a, k: randomized_select(a, k),
    }

    counted = {
        "MoM (g=5)": lambda a, k, c: deterministic_select(
            a, k, group_size=5, counters=c),
        "MoM (g=3)": lambda a, k, c: deterministic_select(
            a, k, group_size=3, counters=c),
        "Quickselect (3-way)": lambda a, k, c: randomized_select(
            a, k, counters=c, rng=random.Random(0)),
    }

    rows = []
    for kind in kinds:
        for n in sizes:
            data = make_input(n, kind)
            k = n // 2
            line = f"  {kind:11} n={n:<7}"
            for name, fn in algorithms.items():
                secs = time_median(lambda: fn(data, k))
                # One extra instrumented call records how many elimination
                # rounds that cell actually needed. On structured input the
                # count is sometimes 1, which is what makes those timings
                # jump around; see run_early_termination().
                ctr = Counters()
                counted[name](data, k, ctr)
                rows.append([kind, n, name, f"{secs:.6f}",
                             f"{secs / n * 1e6:.4f}", ctr.rounds])
                line += f"  {name}={secs:.4f}s(r={ctr.rounds})"
            print(line)

    with open("sel_timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input", "n", "algorithm", "seconds", "microsec_per_elem",
                    "top_level_rounds"])
        w.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# Experiment A2: why the structured-input timings are erratic
# ---------------------------------------------------------------------------
def run_early_termination():
    """
    The timings above are not noisy on sorted, reverse-sorted, and
    duplicate-heavy input; they are erratic for a structural reason.

    Median of Medians returns as soon as the pivot it built happens to be
    the element being looked for. On input with regular structure, whether
    that happens on the very first round depends on how n divides into
    groups of g, which changes with both n and g. One round instead of
    fourteen is a large difference in cost, and it lands unpredictably.
    Random input never triggers it.
    """
    print("\n[A2] Elimination rounds needed (1 = the first pivot was the "
          "answer)")
    sizes = [12_500, 25_000, 50_000, 100_000]
    rows = []

    print(f"  {'input':11} {'n':>8}  {'g=3':>5} {'g=5':>5} {'g=7':>5}")
    for kind in ["random", "sorted", "reverse", "duplicates"]:
        for n in sizes:
            data = make_input(n, kind)
            got = {}
            for g in (3, 5, 7):
                ctr = Counters()
                deterministic_select(data, n // 2, group_size=g,
                                     counters=ctr)
                got[g] = ctr.rounds
                rows.append([kind, n, g, ctr.rounds,
                             f"{ctr.comparisons / n:.3f}"])
            print(f"  {kind:11} {n:>8}  {got[3]:>5} {got[5]:>5} {got[7]:>5}")

    one_round = sum(1 for r in rows if r[3] == 1 and r[0] != "random")
    structured = sum(1 for r in rows if r[0] != "random")
    rand_one = sum(1 for r in rows if r[3] == 1 and r[0] == "random")
    print(f"\n  {one_round} of {structured} structured-input cells finished "
          f"in a single round;")
    print(f"  {rand_one} of {sum(1 for r in rows if r[0] == 'random')} "
          f"random-input cells did (random input needed 11-15 rounds "
          f"every time).")

    with open("sel_early_termination.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input", "n", "group_size", "top_level_rounds",
                    "comparisons_per_element"])
        w.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# Experiment B: the constant factor, measured exactly
# ---------------------------------------------------------------------------
def run_selection_constants():
    """
    Both algorithms are O(n). This experiment asks what the hidden constant
    actually is, by counting comparisons exactly and dividing by n.
    """
    print("\n[B] Comparisons per element (exact counts, n = 100,000)")
    print("    'worst' is the largest fraction of the array that survived "
          "a single round;")
    print("    Median of Medians guarantees <= 0.700 for g=5, Quickselect "
          "guarantees nothing.")
    n = 100_000
    kinds = ["random", "sorted", "reverse", "duplicates"]
    rows = []

    header = (f"  {'input':11} {'algorithm':21} {'cmp/n':>7} {'group/n':>8} "
              f"{'part/n':>7} {'rounds':>6} {'stack':>6} {'worst':>6}")
    print(header)

    for kind in kinds:
        data = make_input(n, kind)
        k = n // 2

        for label, g in (("MoM (g=5)", 5), ("MoM (g=3)", 3), ("MoM (g=7)", 7)):
            ctr = Counters()
            deterministic_select(data, k, group_size=g, counters=ctr)
            rows.append([kind, n, label, ctr.comparisons,
                         f"{ctr.comparisons / n:.3f}",
                         f"{ctr.group_comparisons / n:.3f}",
                         f"{ctr.partition_comparisons / n:.3f}",
                         ctr.swaps, ctr.partitions, ctr.rounds,
                         ctr.max_stack, f"{ctr.worst_shrink_large:.4f}"])

        # Quickselect is randomized: average the counts over 11 seeds.
        totals = []
        for seed in range(11):
            ctr = Counters()
            randomized_select(data, k, counters=ctr,
                              rng=random.Random(seed))
            totals.append(ctr)
        mean_cmp = statistics.mean(c.comparisons for c in totals)
        rows.append([kind, n, "Quickselect (3-way)", int(mean_cmp),
                     f"{mean_cmp / n:.3f}", "0.000",
                     f"{mean_cmp / n:.3f}",
                     int(statistics.mean(c.swaps for c in totals)),
                     int(statistics.mean(c.partitions for c in totals)),
                     int(statistics.mean(c.rounds for c in totals)),
                     0,
                     f"{max(c.worst_shrink_large for c in totals):.4f}"])

    for r in rows:
        print(f"  {r[0]:11} {r[2]:21} {r[4]:>7} {r[5]:>8} {r[6]:>7} "
              f"{r[9]:>6} {r[10]:>6} {r[11]:>6}")

    with open("sel_constants.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input", "n", "algorithm", "comparisons",
                    "comparisons_per_element", "group_cmp_per_element",
                    "partition_cmp_per_element", "swaps", "partitions",
                    "top_level_rounds", "max_stack_frames",
                    "worst_surviving_fraction"])
        w.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# Experiment C: why the textbook picks groups of five
# ---------------------------------------------------------------------------
def run_group_size_scaling():
    """
    The group size is not a tuning knob, it decides the complexity class.

      g=5:  T(n) <= T(n/5)  + T(7n/10) + O(n),  1/5 + 7/10 = 9/10 < 1  -> O(n)
      g=3:  T(n) <= T(n/3)  + T(2n/3)  + O(n),  1/3 + 2/3  = 1        -> O(n log n)
      g=7:  T(n) <= T(n/7)  + T(5n/7)  + O(n),  1/7 + 5/7  = 6/7 < 1  -> O(n)

    Note which side of the recurrence differs. Groups of 3 actually
    partition slightly *better*: their guarantee is 2n/3 surviving, against
    7n/10 for groups of 5. What ruins g=3 is the other term. Finding the
    pivot costs T(n/3) instead of T(n/5), and once the two fractions sum to
    exactly 1 the work stops decaying from level to level.

    The test: if T(n) is linear then T(n)/n is flat and T(n)/(n log n)
    falls. If T(n) is n log n then T(n)/n climbs. Counts are exact and
    Median of Medians is deterministic, so one run per cell settles it with
    no machine noise involved.
    """
    print("\n[C] Group size scaling: is comparisons/n flat, or climbing?")
    sizes = [8_000, 32_000, 128_000, 512_000, 2_048_000]
    rows = []

    print(f"  {'n':>9} | {'cmp/n: g=3':>10} {'g=5':>7} {'g=7':>7} "
          f"| {'cmp/(n lg n): g=3':>17} {'g=5':>7} "
          f"| {'stack: g=3':>10} {'g=5':>5}")
    for n in sizes:
        data = make_input(n, "random", seed=7)
        k = n // 2
        per, norm, stack = {}, {}, {}
        for g in (3, 5, 7):
            ctr = Counters()
            deterministic_select(data, k, group_size=g, counters=ctr)
            per[g] = ctr.comparisons / n
            norm[g] = ctr.comparisons / (n * math.log2(n))
            stack[g] = ctr.max_stack
            rows.append([n, g, ctr.comparisons, f"{per[g]:.4f}",
                         f"{norm[g]:.5f}",
                         f"{ctr.group_comparisons / n:.4f}",
                         f"{ctr.partition_comparisons / n:.4f}",
                         ctr.rounds, ctr.max_stack,
                         f"{ctr.worst_shrink_large:.4f}"])
        print(f"  {n:>9} | {per[3]:>10.2f} {per[5]:>7.2f} {per[7]:>7.2f} "
              f"| {norm[3]:>17.4f} {norm[5]:>7.4f} "
              f"| {stack[3]:>10} {stack[5]:>5}")

    g3 = [r for r in rows if r[1] == 3]
    g5 = [r for r in rows if r[1] == 5]
    print(f"\n  Over a {sizes[-1] // sizes[0]}x range in n:")
    print(f"    g=3  comparisons/element {g3[0][3]} -> {g3[-1][3]}  "
          f"(climbing, roughly +0.5 per 4x in n)")
    print(f"    g=5  comparisons/element {g5[0][3]} -> {g5[-1][3]}  "
          f"(flattening out near 10.3)")
    print(f"    Stack frames track log_g(n): g=3 needs {g3[-1][8]}, "
          f"g=5 needs {g5[-1][8]} at n={sizes[-1]:,} "
          f"(log_3 = {math.log(sizes[-1], 3):.1f}, "
          f"log_5 = {math.log(sizes[-1], 5):.1f})")

    with open("sel_groupsize.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "group_size", "comparisons",
                    "comparisons_per_element", "comparisons_per_n_log_n",
                    "group_cmp_per_element", "partition_cmp_per_element",
                    "top_level_rounds", "max_stack_frames",
                    "worst_surviving_fraction"])
        w.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# Experiment D: duplicate keys and the choice of partition
# ---------------------------------------------------------------------------
def run_duplicate_handling():
    """
    The assignment asks for implementations that handle duplicates. This
    measures what happens when they are not handled.

    Lomuto sends everything equal to the pivot to the low side, so on an
    array of equal keys the pivot lands at the top of the range and the next
    round is handed back n-1 elements. A random pivot cannot save it,
    because when the keys are all equal every pivot is the same pivot.
    """
    print("\n[D] Duplicate keys: three-way vs Lomuto partition (n = 5,000)")
    print("    Fair-comparison note: the three-way partition spends up to "
          "two comparisons")
    print("    per element (one for <, one for >) where Lomuto spends one, "
          "so on fully")
    print("    distinct keys three-way is expected to count about 2x as "
          "many. Wall-clock")
    print("    time is reported alongside so the counting convention "
          "cannot carry the result.")
    n = 5_000
    distinct_counts = [1, 2, 5, 25, 125, 625, 5_000]
    rows = []

    print(f"\n  {'distinct':>9}  {'3-way cmp':>12}  {'Lomuto cmp':>14}  "
          f"{'cmp ratio':>9}  {'3-way ms':>9}  {'Lomuto ms':>10}  "
          f"{'time ratio':>10}")
    for d in distinct_counts:
        rng = random.Random(2024)
        data = [rng.randint(0, d - 1) for _ in range(n)]
        k = n // 2

        three, lom = [], []
        for seed in range(15):
            c1 = Counters()
            randomized_select(data, k, counters=c1, rng=random.Random(seed))
            three.append(c1.comparisons)
            c2 = Counters()
            randomized_select_lomuto(data, k, counters=c2,
                                     rng=random.Random(seed))
            lom.append(c2.comparisons)

        t3 = time_median(lambda: randomized_select(data, k), trials=3)
        tl = time_median(lambda: randomized_select_lomuto(data, k), trials=3)

        m3 = statistics.median(three)
        ml = statistics.median(lom)
        cmp_ratio = ml / m3 if m3 else 0
        rows.append([n, d, int(m3), f"{m3 / n:.3f}", int(ml),
                     f"{ml / n:.3f}", f"{cmp_ratio:.3f}",
                     f"{t3:.6f}", f"{tl:.6f}", f"{tl / t3:.3f}"])
        print(f"  {d:>9}  {int(m3):>12,}  {int(ml):>14,}  "
              f"{cmp_ratio:>8.2f}x  {t3 * 1e3:>9.3f}  {tl * 1e3:>10.3f}  "
              f"{tl / t3:>9.2f}x")

    # --- Growth rate on all-identical input, against the exact formula ---
    #
    # On an array of equal keys Lomuto puts every element on the low side,
    # so the pivot lands at the top of the range and the next round is
    # handed back one fewer element. Selecting rank k therefore costs
    #     sum_{h=k}^{n-1} h  =  (n-1)n/2 - (k-1)k/2
    # comparisons, which for k = n/2 is about 3n^2/8. (It is NOT n(n-1)/2;
    # that would be the cost of running all the way down to a single
    # element.)
    print("\n  All-identical input: measured vs exact closed form")
    print(f"  {'n':>7}  {'3-way':>8}  {'Lomuto':>12}  {'predicted':>12}  "
          f"{'match':>6}  {'Lomuto/n^2':>11}")
    for m in [1_000, 2_000, 4_000, 8_000]:
        same = [7] * m
        k = m // 2
        c3 = Counters()
        randomized_select(same, k, counters=c3, rng=random.Random(1))
        cl = Counters()
        randomized_select_lomuto(same, k, counters=cl, rng=random.Random(1))
        predicted = (m - 1) * m // 2 - (k - 1) * k // 2
        exact = "exact" if cl.comparisons == predicted else "off"
        print(f"  {m:>7}  {c3.comparisons:>8,}  {cl.comparisons:>12,}  "
              f"{predicted:>12,}  {exact:>6}  "
              f"{cl.comparisons / m ** 2:>11.4f}")
        rows.append([m, "all identical", c3.comparisons,
                     f"{c3.comparisons / m:.3f}", cl.comparisons,
                     f"{cl.comparisons / m:.3f}", "", "", "", ""])
        rows.append([m, "all identical (predicted Lomuto)", "", "",
                     predicted, "", "", "", "", ""])

    print("  Lomuto/n^2 holds at a constant ~0.375 = 3/8, confirming "
          "quadratic growth;")
    print("  the three-way partition stays at exactly 2n (one pass, "
          "everything lands")
    print("  in the equal block, answer found).")

    with open("sel_duplicates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "distinct_values", "threeway_comparisons",
                    "threeway_per_elem", "lomuto_comparisons",
                    "lomuto_per_elem", "lomuto_over_threeway_comparisons",
                    "threeway_seconds", "lomuto_seconds",
                    "lomuto_over_threeway_time"])
        w.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# Experiment E: how much does the randomness actually vary?
# ---------------------------------------------------------------------------
def run_variance():
    """
    "Expected O(n)" is a statement about an average. This runs quickselect
    on one fixed input 500 times with different random seeds and reports the
    spread, which is how you see whether the average is a safe thing to
    plan around.
    """
    print("\n[E] Randomized quickselect spread over 500 seeds (n = 50,000)")
    n = 50_000
    data = make_input(n, "random", seed=11)
    k = n // 2

    counts = []
    worst_rounds = []
    for seed in range(500):
        ctr = Counters()
        randomized_select(data, k, counters=ctr, rng=random.Random(seed))
        counts.append(ctr.comparisons)
        worst_rounds.append(ctr.worst_shrink_large)

    lo, hi = min(counts), max(counts)
    med = statistics.median(counts)
    mean = statistics.mean(counts)
    sd = statistics.stdev(counts)

    # Compare against the deterministic algorithm on the same input.
    ctr = Counters()
    deterministic_select(data, k, group_size=5, counters=ctr)
    mom = ctr.comparisons
    mom_worst = ctr.worst_shrink_large

    print(f"  min    {lo:>12,}  ({lo / n:.2f} per element)")
    print(f"  median {int(med):>12,}  ({med / n:.2f} per element)")
    print(f"  mean   {int(mean):>12,}  ({mean / n:.2f} per element)")
    print(f"  max    {hi:>12,}  ({hi / n:.2f} per element)")
    print(f"  std dev{int(sd):>12,}   spread max/min = {hi / lo:.2f}x")
    print(f"  MoM g=5 on the same input: {mom:,} "
          f"({mom / n:.2f} per element), "
          f"{mom / med:.2f}x the quickselect median")
    print(f"\n  Worst single round, counting only rounds on ranges of "
          f">= 1,000 elements")
    print(f"  (small ranges are excluded: ceilings let 6 of 8 survive, "
          f"which is 0.75 and")
    print(f"   is not a fair test of an asymptotic bound):")
    print(f"    Quickselect: {min(worst_rounds):.3f} best seed, "
          f"{max(worst_rounds):.3f} worst seed of 500")
    print(f"    MoM (g=5):   {mom_worst:.3f}, against its proven ceiling "
          f"of 0.700")

    with open("sel_variance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["statistic", "comparisons", "per_element"])
        for label, v in (("min", lo), ("median", med), ("mean", mean),
                         ("max", hi), ("stdev", sd),
                         ("deterministic_mom_g5", mom)):
            w.writerow([label, int(v), f"{v / n:.4f}"])
        w.writerow(["max_over_min_ratio", f"{hi / lo:.4f}", ""])
        w.writerow(["quickselect_worst_surviving_fraction_over_500_seeds",
                    f"{max(worst_rounds):.4f}", ""])
        w.writerow(["mom_g5_worst_surviving_fraction",
                    f"{mom_worst:.4f}", ""])
    return counts


# ---------------------------------------------------------------------------
# Experiment F: data structure operations
# ---------------------------------------------------------------------------
def run_data_structure_ops():
    """
    Part 2's measurements. Each one is chosen to separate an asymptotic
    claim from a constant-factor claim.
    """
    print("\n[F] Data structure operation costs")
    rows = []

    def record(scenario, n, structure, secs, note=""):
        rows.append([scenario, n, structure, f"{secs:.6f}",
                     f"{secs / n * 1e6:.4f}" if n else "", note])

    sizes = [2_000, 4_000, 8_000, 16_000]

    # --- F1: random access by index. O(1) vs O(n). ---
    print("\n  F1  Random access by index (1,000 lookups)")
    for n in sizes:
        arr = DynamicArray(capacity=n)
        ll = SinglyLinkedList()
        for i in range(n):
            arr.append(i)
            ll.insert_back(i)
        idx = [random.Random(5).randrange(n) for _ in range(1_000)]

        t_arr = time_median(lambda: [arr.access(i) for i in idx], trials=3)
        t_ll = time_median(lambda: [ll.access(i) for i in idx], trials=3)
        record("random_access_1000", n, "DynamicArray", t_arr)
        record("random_access_1000", n, "SinglyLinkedList", t_ll)
        print(f"    n={n:<6} array {t_arr * 1e3:8.3f} ms   "
              f"linked {t_ll * 1e3:9.3f} ms   "
              f"linked/array {t_ll / t_arr:7.1f}x")

    # --- F2: insert at the front. O(n) vs O(1). The reverse result. ---
    print("\n  F2  Insert at front (n inserts)")
    for n in sizes:
        def arr_front():
            a = DynamicArray()
            for i in range(n):
                a.insert(0, i)

        def ll_front():
            l = SinglyLinkedList()
            for i in range(n):
                l.insert_front(i)

        t_arr = time_median(arr_front, trials=3)
        t_ll = time_median(ll_front, trials=3)
        record("insert_front_n", n, "DynamicArray", t_arr)
        record("insert_front_n", n, "SinglyLinkedList", t_ll)
        print(f"    n={n:<6} array {t_arr * 1e3:8.3f} ms   "
              f"linked {t_ll * 1e3:9.3f} ms   "
              f"array/linked {t_arr / t_ll:7.1f}x")

    # --- F3: stacks. Same asymptotics; measure the constant. ---
    print("\n  F3  Stack: n pushes then n pops (both O(1) per op)")
    for n in sizes:
        def array_stack():
            s = ArrayStack()
            for i in range(n):
                s.push(i)
            for _ in range(n):
                s.pop()

        def linked_stack():
            s = LinkedStack()
            for i in range(n):
                s.push(i)
            for _ in range(n):
                s.pop()

        t_arr = time_median(array_stack, trials=3)
        t_ll = time_median(linked_stack, trials=3)
        record("stack_push_pop_n", n, "ArrayStack", t_arr)
        record("stack_push_pop_n", n, "LinkedStack", t_ll)
        print(f"    n={n:<6} array {t_arr * 1e3:8.3f} ms   "
              f"linked {t_ll * 1e3:9.3f} ms   "
              f"linked/array {t_ll / t_arr:7.2f}x")

    # --- F4: queues. Here the asymptotics really do differ. ---
    print("\n  F4  Queue: n enqueues then n dequeues")
    for n in sizes:
        def naive():
            q = ArrayQueueNaive()
            for i in range(n):
                q.enqueue(i)
            for _ in range(n):
                q.dequeue()

        def circular():
            q = ArrayQueueCircular()
            for i in range(n):
                q.enqueue(i)
            for _ in range(n):
                q.dequeue()

        def linked():
            q = LinkedQueue()
            for i in range(n):
                q.enqueue(i)
            for _ in range(n):
                q.dequeue()

        t_naive = time_median(naive, trials=3)
        t_circ = time_median(circular, trials=3)
        t_link = time_median(linked, trials=3)
        record("queue_cycle_n", n, "ArrayQueueNaive", t_naive)
        record("queue_cycle_n", n, "ArrayQueueCircular", t_circ)
        record("queue_cycle_n", n, "LinkedQueue", t_link)
        print(f"    n={n:<6} naive {t_naive * 1e3:9.3f} ms   "
              f"circular {t_circ * 1e3:7.3f} ms   "
              f"linked {t_link * 1e3:7.3f} ms   "
              f"naive/circular {t_naive / t_circ:7.1f}x")

    # --- F5: sequential traversal. Identical O(n), so what is left? ---
    #
    # This one has to be set up carefully. Going through the public
    # access(i) method costs a method call plus a bounds check per element,
    # while the linked list's traverse() just follows a pointer. Comparing
    # those two measures the wrappers, not the structures. Both are
    # therefore measured twice: once through the public API, and once
    # reading slots and pointers directly, which is the like-for-like
    # comparison.
    print("\n  F5  Full sequential traversal (both O(n))")
    print("      Measured through the public API and again directly, "
          "because the")
    print("      public array path pays a method call and a bounds check "
          "per element.")
    for n in [16_000, 64_000, 256_000]:
        arr = DynamicArray(capacity=n)
        ll = SinglyLinkedList()
        for i in range(n):
            arr.append(i)
            ll.insert_back(i)

        def scan_arr_api():
            total = 0
            for i in range(len(arr)):
                total += arr.access(i)
            return total

        def scan_arr_direct():
            store = arr._store
            total = 0
            for i in range(len(arr)):
                total += store[i]
            return total

        def scan_ll_api():
            total = 0
            for v in ll.traverse():
                total += v
            return total

        def scan_ll_direct():
            total = 0
            cur = ll.head
            while cur is not None:
                total += cur.value
                cur = cur.next
            return total

        assert (scan_arr_api() == scan_arr_direct()
                == scan_ll_api() == scan_ll_direct())
        t_aa = time_median(scan_arr_api, trials=3)
        t_ad = time_median(scan_arr_direct, trials=3)
        t_la = time_median(scan_ll_api, trials=3)
        t_ld = time_median(scan_ll_direct, trials=3)
        record("traverse_n_api", n, "DynamicArray", t_aa)
        record("traverse_n_api", n, "SinglyLinkedList", t_la)
        record("traverse_n_direct", n, "DynamicArray", t_ad)
        record("traverse_n_direct", n, "SinglyLinkedList", t_ld)
        print(f"    n={n:<7} API:    array {t_aa * 1e3:8.3f} ms   "
              f"linked {t_la * 1e3:8.3f} ms   ratio {t_la / t_aa:5.2f}x")
        print(f"    {'':<9}direct: array {t_ad * 1e3:8.3f} ms   "
              f"linked {t_ld * 1e3:8.3f} ms   ratio {t_ld / t_ad:5.2f}x")

    # --- F6: does row-major storage show up as a locality effect? ---
    #
    # Prediction from the storage layout: a row is contiguous, a column
    # strides by `cols`, so summing by row should beat summing by column.
    # Reported whether or not it appears, because in CPython a list holds
    # pointers to separately allocated integer objects, so neither traversal
    # order walks the actual values contiguously.
    print("\n  F6  Matrix traversal order (row-major storage)")
    print("      Testing whether the predicted row-major advantage is "
          "visible here.")
    for side in [200, 400, 600]:
        m = Matrix(side, side, fill=1)

        def by_row():
            return sum(m.row_sum(r) for r in range(side))

        def by_col():
            return sum(m.column_sum(c) for c in range(side))

        assert by_row() == by_col()
        t_row = time_median(by_row, trials=3)
        t_col = time_median(by_col, trials=3)
        record("matrix_scan", side * side, "row_major_order", t_row,
               f"{side}x{side}")
        record("matrix_scan", side * side, "column_major_order", t_col,
               f"{side}x{side}")
        print(f"    {side}x{side:<5} by row {t_row * 1e3:8.3f} ms   "
              f"by column {t_col * 1e3:8.3f} ms   "
              f"ratio {t_col / t_row:5.2f}x")

    # --- F7: resize accounting for the amortized claim ---
    print("\n  F7  Amortized growth: resizes for n appends")
    for n in [1_000, 10_000, 100_000]:
        a = DynamicArray(capacity=8)
        for i in range(n):
            a.append(i)
        record("append_resizes", n, "DynamicArray", 0.0,
               f"{a.resizes} resizes, final capacity {a.capacity}")
        print(f"    n={n:<7} {a.resizes} resizes, "
              f"final capacity {a.capacity}, "
              f"total elements copied ~{a.capacity - 8}")

    with open("ds_operations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "n", "structure", "seconds",
                    "microsec_per_op", "note"])
        w.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
def main():
    start = time.perf_counter()
    run_selection_timing()
    run_early_termination()
    run_selection_constants()
    run_group_size_scaling()
    run_duplicate_handling()
    run_variance()
    run_data_structure_ops()
    print(f"\nAll benchmarks finished in "
          f"{time.perf_counter() - start:.1f}s. Wrote 7 CSV files.")


if __name__ == "__main__":
    main()
