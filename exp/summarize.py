#!/usr/bin/env python3
"""Summarize exp/logs/<tag>/ into a Fig 18-style table (and Table 4 page sizes).

Performance = cycles(s64k) / cycles(config)  (higher is better, S-64KB = 1.0)
Remote      = "remote access ratio" from the MCM stat block (line in Fig 18)

Writes exp/logs/<tag>/summary.csv and prints a table. CLAP page-size
selections are listed if the "Malloc %d Selected %d" debug print in
src/cuda-sim/cuda-sim.cc (page_size_selection) is enabled.

    ./summarize.py --tag smoke
"""
import argparse
import csv
import math
import os
import re

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ORDER = ["s64k", "s2m", "clap", "clapmig", "ideal"]

PATTERNS = {
    "cycles": re.compile(r"^gpu_tot_sim_cycle = (\d+)", re.M),
    "insn": re.compile(r"^gpu_tot_sim_insn = (\d+)", re.M),
    "remote": re.compile(r"^remote access ratio\s*= ([\d.]+)", re.M),
    "pw_remote": re.compile(r"^pw remote access ratio\s*= ([\d.]+)", re.M),
    "l2tlb_miss": re.compile(r"^l2 tlb miss rate\s*= ([\d.]+)", re.M),
    "pw_lat": re.compile(r"^page walks avg latency\s*= ([\d.]+)", re.M),
}
SELECT = re.compile(r"Malloc (\d+) Selected (\d+)")


def size_str(pages4k):
    if pages4k <= 1:
        return "OLP"  # profiling failed -> opportunistic large paging
    kb = pages4k * 4
    return f"{kb // 1024}MB" if kb >= 1024 else f"{kb}KB"


def parse(log):
    with open(log, errors="replace") as f:
        text = f.read()
    r = {}
    for k, p in PATTERNS.items():
        m = p.findall(text)
        r[k] = float(m[-1]) if m else None  # last = cumulative at exit
    r["selected"] = {int(a): int(b) for a, b in SELECT.findall(text)}
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()
    root = os.path.join(EXP_DIR, "logs", args.tag)
    if not os.path.isdir(root):
        raise SystemExit(f"no logs at {root}")

    rows = []
    for wl in sorted(os.listdir(root)):
        wdir = os.path.join(root, wl)
        if not os.path.isdir(wdir):
            continue
        res = {}
        for cfg in sorted(os.listdir(wdir), key=lambda c: ORDER.index(c) if c in ORDER else 99):
            log = os.path.join(wdir, cfg, "sim.log")
            if os.path.exists(log):
                st_f = os.path.join(wdir, cfg, "status")
                res[cfg] = parse(log)
                res[cfg]["status"] = open(st_f).read().strip() if os.path.exists(st_f) else "?"
        base = res.get("s64k", {}).get("cycles")
        for cfg, r in res.items():
            perf = base / r["cycles"] if base and r["cycles"] else None
            sel = " ".join(f"m{m}:{size_str(s)}" for m, s in sorted(r["selected"].items()))
            rows.append(dict(workload=wl, config=cfg, status=r["status"], cycles=r["cycles"],
                             insn=r["insn"], perf_vs_s64k=perf, remote_ratio=r["remote"],
                             pw_remote_ratio=r["pw_remote"], l2tlb_miss_rate=r["l2tlb_miss"],
                             pw_avg_latency=r["pw_lat"], clap_selected=sel))

    out = os.path.join(root, "summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["workload"])
        w.writeheader()
        w.writerows(rows)

    fmt = lambda v, s="{:.3f}": "-" if v is None else s.format(v)
    print(f"{'workload':14s} {'config':6s} {'status':10s} {'cycles':>12s} {'perf':>6s} {'remote':>7s} {'L2TLBmiss':>9s}  CLAP selected")
    for r in rows:
        print(f"{r['workload']:14s} {r['config']:6s} {r['status']:10s} {fmt(r['cycles'], '{:.0f}'):>12s} "
              f"{fmt(r['perf_vs_s64k']):>6s} {fmt(r['remote_ratio']):>7s} {fmt(r['l2tlb_miss_rate']):>9s}  {r['clap_selected']}")

    # Geomean of perf per config over workloads that have it (Fig 18 'Gmean' bar)
    print()
    for cfg in ORDER:
        vals = [r["perf_vs_s64k"] for r in rows if r["config"] == cfg and r["perf_vs_s64k"]]
        if vals:
            g = math.exp(sum(map(math.log, vals)) / len(vals))
            print(f"Gmean {cfg:6s} {g:.3f}  (n={len(vals)})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
