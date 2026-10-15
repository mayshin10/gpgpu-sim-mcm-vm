#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib"]
# ///
"""Plot a CLAP (MICRO'25) Fig. 20-style figure from exp/logs/<tag>/<workload>/ (default GEMM_REUSE).

Bars: performance normalized to the first config present (S-4KB; S-64KB for tags without it).
Line: remote access ratio ("remote access ratio" from the MCM stat block, whole run).

    uv run plot_fig20.py --tag pm_gemm_full
writes exp/results/fig20_<tag>.png and exp/results/fig20_<tag>.csv (the plotted
values plus per-kernel cycles and migration counts).
"""
import argparse
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
# (config dir, x label, fill, hatch) - grayscale like the paper
BARS = [
    ("s4k", "S-4KB", "#ffffff", ""),
    ("s64k", "S-64KB", "#f0f0f0", ""),
    ("s2m", "S-2MB", "#d0d0d0", ""),
    ("clap", "CLAP", "#a0a0a0", ""),
    ("clapmig", "CLAP+\nmigration", "#505050", ""),
]
CYCLES = re.compile(r"^gpu_tot_sim_cycle = (\d+)", re.M)
KCYCLES = re.compile(r"^gpu_sim_cycle = (\d+)", re.M)
MIGRATIONS = re.compile(r"^page_migration_completed = (\d+)", re.M)
# remote-caching runs: rc_req_side_* (-enable_mem_side 0), rc_res_side_* (1)
TITLES = {"req_side": "Both the requester's and the owner's L2", "res_side": "Owner's L2 only"}
REMOTE = re.compile(r"^remote access ratio\s*= ([\d.]+)", re.M)


def parse(log):
    with open(log, errors="replace") as f:
        text = f.read()
    cyc, rem = CYCLES.findall(text), REMOTE.findall(text)
    if not cyc or not rem:
        raise SystemExit(f"missing stats in {log}")
    mig = MIGRATIONS.findall(text)
    kcyc = [int(c) for c in KCYCLES.findall(text)]  # per kernel launch
    # last = cumulative at exit
    return int(cyc[-1]), float(rem[-1]), kcyc, int(mig[-1]) if mig else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="pm_gemm_full")
    ap.add_argument("--workload", default="GEMM_REUSE")
    args = ap.parse_args()

    wdir = os.path.join(EXP_DIR, "logs", args.tag, args.workload)
    bars = [b for b in BARS if os.path.exists(os.path.join(wdir, b[0], "sim.log"))]
    stats = {cfg: parse(os.path.join(wdir, cfg, "sim.log")) for cfg, *_ in bars}
    base_cfg = bars[0][0]
    base = stats[base_cfg][0]
    perf = [base / stats[cfg][0] for cfg, *_ in bars]
    remote = [stats[cfg][1] for cfg, *_ in bars]

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(3.6, 2.1))
    x = range(len(bars))
    ax.bar(x, perf, width=0.6, color=[b[2] for b in bars],
           edgecolor="black", linewidth=0.8, zorder=2)
    for xi, p, b in zip(x, perf, bars):  # value inside the bar top, clear of the line
        ink = "white" if b[2] in ("#505050",) else "black"
        ax.text(xi, p - 0.03, f"{p:.2f}", ha="center", va="top", fontsize=7.5, color=ink)
    top = max(1.8, max(perf) * 1.1)
    ax.set_ylim(0.5, top)
    ax.set_yticks([t / 4 for t in range(2, int(top * 4) + 1)])
    ax.set_ylabel("Performance")
    for key, title in TITLES.items():
        if key in args.tag:
            ax.set_title(title, fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([b[1] for b in bars], rotation=45, ha="right", rotation_mode="anchor")
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.plot(list(x), remote, "--o", color="black", linewidth=1.4, markersize=4, zorder=3)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_ylabel("Remote ratio")

    fig.tight_layout()
    out_dir = os.path.join(EXP_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"fig20_{args.tag}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")

    out = os.path.join(out_dir, f"fig20_{args.tag}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "kernel1_cycles", "kernel2_cycles", "total_cycles",
                    f"perf_vs_{base_cfg}", "remote_ratio", "migrations"])
        for (cfg, *_), p, r in zip(bars, perf, remote):
            tot, _, kcyc, mig = stats[cfg]
            w.writerow([cfg, kcyc[0], kcyc[1], tot, f"{p:.3f}", f"{r:.3f}", mig])
    print(f"wrote {out}")

    print(f"\n{'config':10s} {'cycles':>10s} {'perf':>6s} {'remote':>7s}")
    for (cfg, *_), p, r in zip(bars, perf, remote):
        print(f"{cfg:10s} {stats[cfg][0]:>10d} {p:>6.3f} {r:>7.3f}")


if __name__ == "__main__":
    main()
