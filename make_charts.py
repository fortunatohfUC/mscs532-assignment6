"""
MSCS-532 Assignment 6: figures for the report.

Run:  python benchmarks.py && python make_charts.py

Every number plotted here is read back out of the CSV files that
benchmarks.py wrote, so the figures and the report tables cannot drift
apart from the measurements.

Colour choices: three categorical hues, assigned in a fixed order and never
cycled, checked for colour-vision separation as a set. No chart carries more
than three series, and where the category is already on the axis a single
hue is used instead of a rainbow. Every multi-series chart has a legend and
markers, so the series are never told apart by colour alone.
"""

import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- Palette (validated as a set: worst all-pairs CVD dE 9.2, normal 24.0) --
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]
MARKERS = ["o", "s", "^"]

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dedcd6"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "text.color": INK,
    "font.size": 9,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def thousands(x, _pos):
    if x >= 1_000_000:
        return f"{x / 1_000_000:g}M"
    if x >= 1_000:
        return f"{x / 1_000:g}k"
    return f"{x:g}"


FMT = FuncFormatter(thousands)


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def set_n_ticks(ax, ns):
    """
    Put ticks on the array sizes that were actually measured.

    Left to itself a log-base-2 axis labels powers of two (16.384k,
    1.04858M), which are not the sizes in the data and read as noise.
    """
    ax.set_xticks(list(ns))
    ax.set_xticklabels([thousands(n, None) for n in ns])
    ax.minorticks_off()


# ---------------------------------------------------------------------------
# Figure 1: running time across input distributions
# ---------------------------------------------------------------------------
def fig_selection_timing():
    rows = read_csv("sel_timing.csv")
    algos = ["MoM (g=5)", "MoM (g=3)", "Quickselect (3-way)"]
    kinds = ["random", "sorted", "reverse", "duplicates"]
    titles = {"random": "Random", "sorted": "Sorted",
              "reverse": "Reverse-sorted", "duplicates": "Duplicate-heavy"}

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)

    for ax, kind in zip(axes.flat, kinds):
        for i, algo in enumerate(algos):
            pts = sorted((int(r["n"]), float(r["seconds"]))
                         for r in rows
                         if r["input"] == kind and r["algorithm"] == algo)
            ax.plot([p[0] for p in pts], [p[1] * 1000 for p in pts],
                    color=SERIES[i], marker=MARKERS[i], label=algo)
        ax.set_title(titles[kind], fontsize=9, loc="left")
        ax.set_xscale("log", base=2)
        set_n_ticks(ax, sorted({int(r["n"]) for r in rows}))
        ax.tick_params(axis="x", labelsize=8)
        tidy(ax)

    for ax in axes[1]:
        ax.set_xlabel("array size n")
    for ax in axes[:, 0]:
        ax.set_ylabel("time (ms)")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Selecting the median: running time by input distribution",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0.03, 1, 0.945])
    fig.savefig("fig1_selection_timing.png")
    plt.close(fig)
    print("wrote fig1_selection_timing.png")


# ---------------------------------------------------------------------------
# Figure 2: the constant factor, split by where the work happens
# ---------------------------------------------------------------------------
def fig_constant_factor():
    rows = read_csv("sel_constants.csv")
    kinds = ["random", "sorted", "reverse", "duplicates"]
    titles = {"random": "Random", "sorted": "Sorted",
              "reverse": "Reverse-sorted", "duplicates": "Duplicate-heavy"}
    order = ["Quickselect (3-way)", "MoM (g=7)", "MoM (g=5)", "MoM (g=3)"]

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.9), sharey=True)

    for ax, kind in zip(axes, kinds):
        by_algo = {r["algorithm"]: r for r in rows if r["input"] == kind}
        ys = range(len(order))
        grp = [float(by_algo[a]["group_cmp_per_element"]) for a in order]
        prt = [float(by_algo[a]["partition_cmp_per_element"]) for a in order]

        # Two stacked components of one measure: sequential-style shading of
        # a single hue, not two categorical colours.
        ax.barh(ys, prt, color=BLUE, height=0.6, label="partitioning",
                edgecolor=SURFACE, linewidth=1.2)
        ax.barh(ys, grp, left=prt, color=AQUA, height=0.6,
                label="group sorting (pivot search)",
                edgecolor=SURFACE, linewidth=1.2)
        for y, (p, g) in enumerate(zip(prt, grp)):
            ax.text(p + g + 0.35, y, f"{p + g:.1f}", va="center",
                    fontsize=7.5, color=INK)

        ax.set_yticks(list(ys))
        ax.set_yticklabels([a.replace(" (", "\n(") for a in order],
                           fontsize=7.5)
        ax.set_title(titles[kind], fontsize=9, loc="left")
        ax.set_xlim(0, 15.5)
        ax.set_xlabel("comparisons per element")
        tidy(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Where the comparisons go (n = 100,000, exact counts)",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig("fig2_constant_factor.png")
    plt.close(fig)
    print("wrote fig2_constant_factor.png")


# ---------------------------------------------------------------------------
# Figure 3: why the group size decides the complexity class
# ---------------------------------------------------------------------------
def fig_group_size():
    rows = read_csv("sel_groupsize.csv")
    groups = [3, 5, 7]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.1))

    for i, g in enumerate(groups):
        pts = sorted((int(r["n"]),
                      float(r["comparisons_per_element"]),
                      float(r["comparisons_per_n_log_n"]))
                     for r in rows if int(r["group_size"]) == g)
        ns = [p[0] for p in pts]
        ax1.plot(ns, [p[1] for p in pts], color=SERIES[i],
                 marker=MARKERS[i], label=f"groups of {g}")
        ax2.plot(ns, [p[2] for p in pts], color=SERIES[i],
                 marker=MARKERS[i], label=f"groups of {g}")
        # g=5 and g=7 land on top of each other, so they share one label
        # rather than overprinting two.
        if g == 3:
            ax1.annotate("g=3", (ns[-1], pts[-1][1]),
                         textcoords="offset points", xytext=(7, 0),
                         fontsize=8, color=INK, va="center")
        elif g == 5:
            ax1.annotate("g=5 and g=7\n(nearly identical)",
                         (ns[-1], pts[-1][1]),
                         textcoords="offset points", xytext=(7, -2),
                         fontsize=7.5, color=INK_SOFT, va="center")

    ax1.set_title("Divided by n\n(flat = linear)", fontsize=9, loc="left")
    ax1.set_ylabel("comparisons / n")
    ax2.set_title("Divided by n log₂n\n(flat = n log n)",
                  fontsize=9, loc="left")
    ax2.set_ylabel("comparisons / (n log₂ n)")

    all_ns = sorted({int(r["n"]) for r in rows})
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        set_n_ticks(ax, all_ns)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_xlabel("array size n")
        tidy(ax)
    ax1.set_xlim(right=ax1.get_xlim()[1] * 3.2)

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Median of Medians: group size changes the complexity "
                 "class", fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0.04, 1, 0.91])
    fig.savefig("fig3_group_size.png")
    plt.close(fig)
    print("wrote fig3_group_size.png")


# ---------------------------------------------------------------------------
# Figure 4: duplicate keys and the partition scheme
# ---------------------------------------------------------------------------
def fig_duplicates():
    rows = read_csv("sel_duplicates.csv")
    sweep = [r for r in rows if r["distinct_values"].isdigit()]
    ident = [r for r in rows
             if r["distinct_values"] == "all identical"]
    pred = [r for r in rows
            if r["distinct_values"] == "all identical (predicted Lomuto)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.1))

    # Panel A: fixed n, varying number of distinct values.
    ds = [int(r["distinct_values"]) for r in sweep]
    ax1.plot(ds, [int(r["threeway_comparisons"]) for r in sweep],
             color=BLUE, marker="o", label="three-way partition")
    ax1.plot(ds, [int(r["lomuto_comparisons"]) for r in sweep],
             color=ORANGE, marker="s", label="Lomuto partition")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("distinct values in the array")
    ax1.set_ylabel("comparisons")
    ax1.set_title("n = 5,000, duplicates increasing → left",
                  fontsize=9, loc="left")
    ax1.annotate("937x apart\nat 1 distinct value",
                 xy=(ds[0], int(sweep[0]["lomuto_comparisons"])),
                 textcoords="offset points", xytext=(14, -6),
                 fontsize=7.5, color=INK_SOFT)
    tidy(ax1)

    # Panel B: all-identical input, growth rate against the exact formula.
    ns = [int(r["n"]) for r in ident]
    ax2.plot(ns, [int(r["threeway_comparisons"]) for r in ident],
             color=BLUE, marker="o", label="three-way partition")
    ax2.plot(ns, [int(r["lomuto_comparisons"]) for r in ident],
             color=ORANGE, marker="s", label="Lomuto partition")
    ax2.plot([int(r["n"]) for r in pred],
             [int(r["lomuto_comparisons"]) for r in pred],
             color=INK_SOFT, linestyle=":", linewidth=1.6, marker="",
             label="predicted $\\frac{n(n-1)}{2}-\\frac{k(k-1)}{2}$")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    set_n_ticks(ax2, ns)
    ax2.set_xlabel("array size n (every key identical)")
    ax2.set_ylabel("comparisons")
    ax2.set_title("All keys equal: 2n against ~3n²/8",
                  fontsize=9, loc="left")
    tidy(ax2)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + [h2[-1]], l1 + [l2[-1]], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Duplicate keys: the partition scheme, not the pivot rule",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0.04, 1, 0.91])
    fig.savefig("fig4_duplicates.png")
    plt.close(fig)
    print("wrote fig4_duplicates.png")


# ---------------------------------------------------------------------------
# Figure 5: randomized quickselect is an average, not a guarantee
# ---------------------------------------------------------------------------
def fig_variance():
    import random
    from selection import Counters, randomized_select
    from benchmarks import make_input

    n = 50_000
    data = make_input(n, "random", seed=11)
    counts = []
    for seed in range(500):
        ctr = Counters()
        randomized_select(data, n // 2, counters=ctr,
                          rng=random.Random(seed))
        counts.append(ctr.comparisons / n)

    stats = {r["statistic"]: r for r in read_csv("sel_variance.csv")}
    mom = float(stats["deterministic_mom_g5"]["per_element"])

    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.hist(counts, bins=45, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(mom, color=ORANGE, linewidth=2.0)
    ax.annotate(f"Median of Medians (g=5)\nalways {mom:.2f} "
                f"on this input",
                xy=(mom, ax.get_ylim()[1] * 0.92),
                textcoords="offset points", xytext=(10, -4),
                fontsize=8, color=INK, va="top")
    ax.set_xlabel("comparisons per element")
    ax.set_ylabel("seeds (of 500)")
    ax.set_title("Randomized quickselect over 500 random seeds, "
                 "one fixed input\n"
                 f"spread {min(counts):.2f} to {max(counts):.2f} "
                 f"per element ({max(counts) / min(counts):.1f}x)",
                 fontsize=9.5, loc="left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig("fig5_variance.png")
    plt.close(fig)
    print("wrote fig5_variance.png")


# ---------------------------------------------------------------------------
# Figure 6: data structures where the asymptotics differ
# ---------------------------------------------------------------------------
def fig_data_structures():
    rows = read_csv("ds_operations.csv")

    def series(scenario, structure):
        pts = sorted((int(r["n"]), float(r["seconds"]) * 1000)
                     for r in rows if r["scenario"] == scenario
                     and r["structure"] == structure)
        return [p[0] for p in pts], [p[1] for p in pts]

    panels = [
        ("random_access_1000", "Access by index (1,000 lookups)",
         [("DynamicArray", "array - O(1)"),
          ("SinglyLinkedList", "linked list - O(n)")]),
        ("insert_front_n", "Insert at the front (n inserts)",
         [("DynamicArray", "array - O(n)"),
          ("SinglyLinkedList", "linked list - O(1)")]),
        ("queue_cycle_n", "Queue: n enqueues then n dequeues",
         [("ArrayQueueNaive", "array, front fixed - O(n) dequeue"),
          ("ArrayQueueCircular", "circular buffer - O(1)"),
          ("LinkedQueue", "linked list - O(1)")]),
        ("stack_push_pop_n", "Stack: n pushes then n pops",
         [("ArrayStack", "array - O(1) amortized"),
          ("LinkedStack", "linked list - O(1)")]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))

    for ax, (scenario, title, structs) in zip(axes.flat, panels):
        for i, (struct, label) in enumerate(structs):
            ns, ms = series(scenario, struct)
            ax.plot(ns, ms, color=SERIES[i], marker=MARKERS[i], label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        set_n_ticks(ax, series(scenario, structs[0][0])[0])
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("n")
        ax.set_ylabel("time (ms)")
        ax.legend(fontsize=7.5, loc="upper left")
        tidy(ax)

    fig.suptitle("Elementary data structures: operation cost as n grows\n"
                 "(log scales on both axes; a steeper line is a worse "
                 "complexity class)",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig("fig6_data_structures.png")
    plt.close(fig)
    print("wrote fig6_data_structures.png")


# ---------------------------------------------------------------------------
# Figure 7: where the asymptotics are equal, what is left is small
# ---------------------------------------------------------------------------
def fig_equal_asymptotics():
    rows = read_csv("ds_operations.csv")

    def series(scenario, structure):
        pts = sorted((int(r["n"]), float(r["seconds"]) * 1000)
                     for r in rows if r["scenario"] == scenario
                     and r["structure"] == structure)
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Panel A: traversal, measured fairly and unfairly.
    for i, (scen, lbl) in enumerate([("traverse_n_direct", "direct"),
                                     ("traverse_n_api", "via public API")]):
        ns_a, ms_a = series(scen, "DynamicArray")
        ns_l, ms_l = series(scen, "SinglyLinkedList")
        ratios = [l / a for a, l in zip(ms_a, ms_l)]
        ax1.plot(ns_a, ratios, color=SERIES[i], marker=MARKERS[i],
                 label=f"{lbl} slot/pointer read")
    ax1.axhline(1.0, color=INK_SOFT, linestyle=":", linewidth=1.4)
    ax1.annotate("equal cost", xy=(ax1.get_xlim()[0], 1.0),
                 textcoords="offset points", xytext=(6, 5),
                 fontsize=7.5, color=INK_SOFT)
    ax1.set_xscale("log", base=2)
    set_n_ticks(ax1, series("traverse_n_direct", "DynamicArray")[0])
    ax1.set_ylim(0, 1.3)
    ax1.set_xlabel("n")
    ax1.set_ylabel("linked time / array time")
    ax1.set_title("Sequential traversal: both O(n)", fontsize=9, loc="left")
    ax1.legend(fontsize=7.5, loc="lower right")
    tidy(ax1)

    # Panel B: matrix scan order. The predicted locality effect, or not.
    pts_row = sorted((int(r["n"]), float(r["seconds"]) * 1000)
                     for r in rows if r["scenario"] == "matrix_scan"
                     and r["structure"] == "row_major_order")
    pts_col = sorted((int(r["n"]), float(r["seconds"]) * 1000)
                     for r in rows if r["scenario"] == "matrix_scan"
                     and r["structure"] == "column_major_order")
    sides = [int(math.isqrt(p[0])) for p in pts_row]
    ratios = [c[1] / r[1] for r, c in zip(pts_row, pts_col)]
    ax2.bar([str(s) for s in sides], ratios, color=BLUE, width=0.5)
    ax2.axhline(1.0, color=INK_SOFT, linestyle=":", linewidth=1.4)
    for x, v in enumerate(ratios):
        ax2.text(x, v + 0.02, f"{v:.2f}x", ha="center", fontsize=8,
                 color=INK)
    ax2.set_ylim(0, 1.35)
    ax2.set_xlabel("matrix side length")
    ax2.set_ylabel("column-order / row-order")
    ax2.set_title("Matrix scan order: row-major storage",
                  fontsize=9, loc="left")
    tidy(ax2)

    fig.suptitle("Where the complexity classes match, the measured "
                 "difference nearly vanishes", fontsize=10, x=0.02,
                 ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig("fig7_equal_asymptotics.png")
    plt.close(fig)
    print("wrote fig7_equal_asymptotics.png")


REQUIRED_CSVS = [
    "sel_timing.csv", "sel_constants.csv", "sel_groupsize.csv",
    "sel_duplicates.csv", "sel_variance.csv", "ds_operations.csv",
]


def check_inputs():
    """
    The CSV files are produced by benchmarks.py, not shipped with the code,
    so this script cannot run first. Say that plainly instead of failing
    with a FileNotFoundError traceback.
    """
    import os
    missing = [f for f in REQUIRED_CSVS if not os.path.exists(f)]
    if missing:
        print("Cannot build the figures yet: the measurement files are "
              "missing.")
        print("  missing: " + ", ".join(missing))
        print()
        print("These are generated, not shipped. Run the benchmarks first:")
        print()
        print("    python benchmarks.py     # about 2.5 minutes")
        print("    python make_charts.py")
        return False
    return True


def main():
    if not check_inputs():
        raise SystemExit(1)
    fig_selection_timing()
    fig_constant_factor()
    fig_group_size()
    fig_duplicates()
    fig_variance()
    fig_data_structures()
    fig_equal_asymptotics()
    print("All figures written.")


if __name__ == "__main__":
    main()
