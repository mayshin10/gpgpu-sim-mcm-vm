#!/usr/bin/env python3
"""Run CLAP (MICRO'25) reproduction sweeps on gpgpu-sim-mcm-vm.

For each (workload, config) pair this creates
    exp/logs/<tag>/<workload>/<config>/
        gpgpusim.config   base ampere-based config + per-config overrides appended
        cmd.txt           command that was executed
        sim.log           simulator stdout+stderr
        status            done | max_insn | fail(<code>) | running
and runs the workload binary inside that directory (GPGPU-Sim reads
./gpgpusim.config from the CWD). Options appended later in the config file
override earlier ones (the option parser applies tokens in order).

Examples:
    ./run.py --list
    ./run.py -w STE GEMM_VIT -c s64k s2m clap --max-insn 50000000 --tag smoke
    nohup ./run.py --all -j 12 --tag full > logs/full.driver.log 2>&1 &

Runs are started by a memory-aware scheduler (see schedule()); memory usage is
logged every 30 s to exp/logs/<tag>.mem.log and each run's peak RSS is stored
in <run dir>/peak_rss_gb.
"""
import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
import time

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_ROOT = os.path.dirname(EXP_DIR)
BASE_CFG = os.path.join(SIM_ROOT, "configs/mcm-tested-cfgs/ampere-based/gpgpusim.config")
LOG_ROOT = os.path.join(EXP_DIR, "logs")

# Common to every config: FT (first-touch) TB & data arrangement = paper baseline (Sec 3.1).
COMMON = [
    ("-mcm_cta_schedule", "1"),
    ("-mcm_data_schedule", "1"),
    ("-gpgpu_memory_partition_indexing", "5"),
    ("-mapping_granularity", "16"),
    ("-enable_mem_side", "1"),  # 1 = remote data caching OFF
]

# Paper Sec 5 configurations available in this repo (Fig 18 subset).
CONFIGS = {
    # S-64KB: reservation-based demand paging, 64KB pages (Fig 18 normalization base)
    "s64k": [("-vm_config", "0"), ("-enable_mem_reserved", "1"), ("-set_page_size", "16"),
             ("-enable_profiling", "0")],
    # S-2MB: 64KB demand mapping into reserved 2MB frames, promoted when full
    "s2m": [("-vm_config", "0"), ("-enable_mem_reserved", "1"), ("-set_page_size", "512"),
            ("-enable_profiling", "0")],
    # CLAP: PMM(20%) + OLP + MMA + coalesced TLB. set_page_size=512 is the OLP/VA-block size.
    "clap": [("-vm_config", "0"), ("-enable_mem_reserved", "1"), ("-set_page_size", "512"),
             ("-enable_profiling", "1"), ("-mapping_threshold", "0.20")],
    # CLAP + migration (Sec 6, Fig 20): CLAP plus C-NUMA co-location migration of data
    # structures shared across kernels; copies use the inter-chip links
    "clapmig": [("-vm_config", "0"), ("-enable_mem_reserved", "1"), ("-set_page_size", "512"),
                ("-enable_profiling", "1"), ("-mapping_threshold", "0.20"),
                ("-enable_page_migration", "1")],
    # Ideal (approx.): 64KB placement + ideal TLB (paper's Ideal = 64KB placement, 2MB translation)
    "ideal": [("-vm_config", "1"), ("-enable_mem_reserved", "1"), ("-set_page_size", "16"),
              ("-enable_profiling", "0")],
}


def load_workloads(path):
    """TSV: <name>\\t<command>. '#' comments allowed. {exp} expands to the exp dir."""
    wls = {}
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if "\t" not in line:
                sys.exit(f"{path}:{ln}: expected '<name><TAB><command>'")
            name, cmd = line.split("\t", 1)
            wls[name.strip()] = cmd.strip().replace("{exp}", EXP_DIR)
    return wls


def find_sim_lib(explicit):
    if explicit:
        return os.path.abspath(explicit)
    cands = sorted(os.path.dirname(p) for p in
                   glob.glob(os.path.join(SIM_ROOT, "lib/*/cuda-*/release/libcudart.so")))
    if not cands:
        sys.exit("No built libcudart.so under lib/. Build the simulator first.")
    cc = os.environ.get("CC", "gcc")
    try:
        out = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout
        ver = re.search(r"\b(\d+\.\d+\.\d+)\b", out).group(1)
        match = [c for c in cands if f"/gcc-{ver}/" in c]
        if match:
            return match[0]
    except Exception:
        pass
    if len(cands) > 1:
        sys.exit("Multiple simulator builds found; pick one with --lib:\n  " + "\n  ".join(cands))
    return cands[0]


def sim_env(lib_dir):
    cuda = os.environ.get("CUDA_INSTALL_PATH")
    if not cuda or not os.path.isdir(cuda):
        sys.exit("CUDA_INSTALL_PATH is not set to a valid directory.")
    env = dict(os.environ)
    # Same variables setup_environment exports at runtime.
    env["GPGPUSIM_ROOT"] = SIM_ROOT
    env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PATH"] = f"{SIM_ROOT}/bin:{cuda}/bin:" + env.get("PATH", "")
    env["PTX_SIM_USE_PTX_FILE"] = "1.ptx"
    env["PTX_SIM_KERNELFILE"] = "_1.ptx"
    env["CUOBJDUMP_SIM_FILE"] = "jj"
    return env


def write_config(run_dir, cfg, max_insn, extra, base_cfg=BASE_CFG):
    with open(base_cfg) as f:
        base = f.read()
    lines = [f"\n\n### ---- exp/run.py overrides: {cfg} ----"]
    lines += [f"{k} {v}" for k, v in COMMON + CONFIGS[cfg]]
    if max_insn is not None:
        lines.append(f"-gpgpu_max_insn {max_insn}")
    lines += extra
    with open(os.path.join(run_dir, "gpgpusim.config"), "w") as f:
        f.write(base + "\n".join(lines) + "\n")


def classify(run_dir, rc):
    try:
        with open(os.path.join(run_dir, "sim.log"), errors="replace") as f:
            text = f.read()  # break message is printed mid-log, before the final stat dump
    except OSError:
        text = ""
    if "GPGPU-Sim: ** break" in text:  # message can be interleaved with other output
        return "max_insn"  # simulator exits with 1 after printing stats
    if rc == 0:
        return "done"
    return f"fail({rc})"


def meminfo_gb(key="MemAvailable"):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1048576
    return 0.0


def rss_gb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return 0.0


def prepare(job, args):
    """Returns False if the run is already finished (skip)."""
    name, cfg, cmd, run_dir = job
    status_f = os.path.join(run_dir, "status")
    if not args.force and os.path.exists(status_f):
        if open(status_f).read().strip() in ("done", "max_insn"):
            return False
    os.makedirs(run_dir, exist_ok=True)
    write_config(run_dir, cfg, args.max_insn, args.set, args.base_cfg)
    with open(os.path.join(run_dir, "cmd.txt"), "w") as f:
        f.write(cmd + "\n")
    return True


def start(job, env):
    name, cfg, cmd, run_dir = job
    with open(os.path.join(run_dir, "status"), "w") as f:
        f.write("running\n")
    log = open(os.path.join(run_dir, "sim.log"), "w")
    # exec: the shell is replaced by the simulator, so the pid/rusage is the simulator's own
    p = subprocess.Popen("exec " + cmd, shell=True, cwd=run_dir, env=env,
                         stdout=log, stderr=subprocess.STDOUT)
    log.close()
    return p


def finish(job, rc, elapsed, peak):
    run_dir = job[3]
    st = classify(run_dir, rc)
    for fn, val in (("status", st), ("elapsed_sec", f"{elapsed:.0f}"), ("peak_rss_gb", f"{peak:.2f}")):
        with open(os.path.join(run_dir, fn), "w") as f:
            f.write(val + "\n")
    return st


def schedule(jobs, env, args, mem_log):
    """Memory-aware scheduler.

    - start a pending run only if MemAvailable - reserve >= estimated need, where the
      need is the largest RSS seen so far for that workload (any config) * 1.3, or
      --default-need-gb if unknown; runs are started at least --stagger seconds apart.
    - if MemAvailable drops below --critical-gb, the most recently started run is
      killed and requeued; no new run starts until another run finishes.
    """
    pending = list(jobs)
    running = {}  # pid -> [job, popen, t0, peak]
    peak_by_wl = {}
    last_start = 0.0
    hold = False
    last_log = 0.0
    ts = lambda: time.strftime("%H:%M:%S")
    while pending or running:
        now = time.time()
        avail = meminfo_gb()
        for pid, r in running.items():
            g = rss_gb(pid)
            r[3] = max(r[3], g)
            peak_by_wl[r[0][0]] = max(peak_by_wl.get(r[0][0], 0.0), g)

        # reap finished runs
        for pid in list(running):
            try:
                wpid, status, ru = os.wait4(pid, os.WNOHANG)
            except ChildProcessError:
                wpid, status, ru = pid, 0, None
            if wpid == 0:
                continue
            job, p, t0, peak = running.pop(pid)
            if ru is not None:
                peak = max(peak, ru.ru_maxrss / 1048576)
            peak_by_wl[job[0]] = max(peak_by_wl.get(job[0], 0.0), peak)
            rc = os.waitstatus_to_exitcode(status)
            p.returncode = rc  # keep Popen from waiting on a reaped pid
            st = finish(job, rc, now - t0, peak)
            hold = False
            print(f"[{ts()}] {job[0]:6s} {job[1]:5s} {st:10s} {(now - t0) / 60:6.1f} min  peak {peak:5.1f} GB",
                  flush=True)

        # memory emergency: kill + requeue the youngest run
        if avail < args.critical_gb and len(running) > 1:
            pid = max(running, key=lambda k: running[k][2])
            job, p, t0, peak = running.pop(pid)
            p.kill()
            p.wait()
            with open(os.path.join(job[3], "status"), "w") as f:
                f.write("requeued\n")
            pending.insert(0, job)
            hold = True
            print(f"[{ts()}] {job[0]:6s} {job[1]:5s} KILLED+requeued (MemAvailable {avail:.1f} GB < "
                  f"{args.critical_gb} GB)", flush=True)

        # start new runs
        if pending and not hold and len(running) < args.jobs and now - last_start >= args.stagger:
            job = pending[0]
            need = peak_by_wl.get(job[0], 0.0) * 1.3 or args.default_need_gb
            if avail - args.reserve_gb >= need:
                pending.pop(0)
                p = start(job, env)
                running[p.pid] = [job, p, now, 0.0]
                last_start = now
                print(f"[{ts()}] {job[0]:6s} {job[1]:5s} start   (avail {avail:.0f} GB, need ~{need:.0f} GB, "
                      f"running {len(running)}, pending {len(pending)})", flush=True)

        if now - last_log >= 30:
            last_log = now
            per = " ".join(f"{r[0][0]}/{r[0][1]}={rss_gb(pid):.1f}" for pid, r in running.items())
            mem_log.write(f"{ts()} avail={avail:.1f}GB running={len(running)} pending={len(pending)} {per}\n")
            mem_log.flush()
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workloads-file", default=os.path.join(EXP_DIR, "workloads.tsv"))
    ap.add_argument("-w", "--workloads", nargs="+", help="workload names (default: --all required)")
    ap.add_argument("--all", action="store_true", help="run every workload in the file")
    ap.add_argument("-c", "--configs", nargs="+", default=["s64k", "s2m", "clap"], choices=list(CONFIGS))
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--tag", default="default", help="logs go to exp/logs/<tag>/")
    ap.add_argument("--max-insn", type=int, help="override -gpgpu_max_insn (base config: 1e9)")
    ap.add_argument("--set", action="append", default=[], metavar="'-opt val'",
                    help="extra config line appended after overrides (repeatable)")
    ap.add_argument("--base-cfg", default=BASE_CFG,
                    help="base gpgpusim.config (e.g. configs/mcm-tested-cfgs/ampere-based-paper/gpgpusim.config)")
    ap.add_argument("--lib", help="simulator lib dir (default: auto-detect under lib/)")
    ap.add_argument("--force", action="store_true", help="rerun even if already done")
    ap.add_argument("--dry-run", action="store_true", help="only generate run dirs/configs")
    ap.add_argument("--list", action="store_true", help="list workloads and configs")
    ap.add_argument("--reserve-gb", type=float, default=30, help="memory kept free when starting runs")
    ap.add_argument("--critical-gb", type=float, default=12, help="kill+requeue youngest run below this")
    ap.add_argument("--default-need-gb", type=float, default=10, help="assumed need of an unseen workload")
    ap.add_argument("--stagger", type=float, default=60, help="min seconds between run starts")
    args = ap.parse_args()

    wls = load_workloads(args.workloads_file)
    if args.list:
        print("configs:", " ".join(CONFIGS))
        for n, c in wls.items():
            print(f"{n:16s} {c}")
        return
    names = list(wls) if args.all else (args.workloads or [])
    if not names:
        ap.error("give -w NAME ... or --all")
    unknown = [n for n in names if n not in wls]
    if unknown:
        sys.exit(f"unknown workloads: {unknown} (see {args.workloads_file})")

    lib_dir = find_sim_lib(args.lib)
    env = sim_env(lib_dir)
    print(f"simulator lib: {lib_dir}\nlogs: {os.path.join(LOG_ROOT, args.tag)}", flush=True)

    jobs = [(n, c, wls[n], os.path.join(LOG_ROOT, args.tag, n, c)) for n in names for c in args.configs]
    jobs = [j for j in jobs if prepare(j, args)]
    if args.dry_run:
        print(f"{len(jobs)} run dirs prepared")
        return
    with open(os.path.join(LOG_ROOT, f"{args.tag}.mem.log"), "a") as mem_log:
        schedule(jobs, env, args, mem_log)

if __name__ == "__main__":
    main()
