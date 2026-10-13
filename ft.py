#!/usr/bin/env python3
"""
ft.py - frame-time capture & analysis harness for Linux gaming experiments.

Subcommands
  run        launch a command N times under MangoHud logging; save CSVs + system metadata
  summarize  per-run metrics for CSV files / run directories
  compare    A/B comparison of two sets of runs (mean +/- 95% CI across runs)
  plot       frame-time series, histogram, and tail-percentile plots
  synth      generate synthetic frame-time CSVs to test the pipeline

Input formats understood by every analysis command:
  * MangoHud CSV logs (any CSV with a "frametime" column, in ms)
  * plain one-column files of frame times in ms (one value per line)

Dependencies: numpy (required), matplotlib (only for `plot`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Two-sided 95% Student-t critical values for df = 1..30 (falls back to 1.96).
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]

# (key, column header, printf format, higher_is_better)
METRICS = [
    ("avg_fps",         "avg FPS",     "{:.1f}", True),
    ("low1_fps",        "1% low FPS",  "{:.1f}", True),
    ("low01_fps",       "0.1% low FPS", "{:.1f}", True),
    ("p99_ms",          "p99 ms",      "{:.2f}", False),
    ("p999_ms",         "p99.9 ms",    "{:.2f}", False),
    ("max_ms",          "max ms",      "{:.1f}", False),
    ("jitter_ms",       "jitter ms",   "{:.3f}", False),
    ("stutters_per_min", "stutter/min", "{:.1f}", False),
]


# --------------------------------------------------------------------------- loading
def load_frametimes(path, skip_s: float = 0.0, max_s: float | None = None) -> np.ndarray:
    """Return frame times in ms. Drops the first `skip_s` seconds and keeps at most `max_s` after that."""
    lines = Path(path).read_text(errors="replace").splitlines()
    vals: list[float] = []
    for i, line in enumerate(lines):  # MangoHud-style: find the header row naming a frametime column
        cols = [c.strip().lower() for c in line.split(",")]
        if "frametime" in cols:
            idx = cols.index("frametime")
            for row in lines[i + 1:]:
                parts = row.split(",")
                try:
                    vals.append(float(parts[idx]))
                except (ValueError, IndexError):
                    continue
            break
    else:  # fallback: first numeric column of each line
        for line in lines:
            try:
                vals.append(float(line.split(",")[0]))
            except ValueError:
                continue
    ft = np.asarray(vals, dtype=float)
    ft = ft[np.isfinite(ft) & (ft > 0)]
    if skip_s > 0 or max_s is not None:
        t = np.cumsum(ft) / 1000.0
        keep = t > skip_s
        if max_s is not None:
            keep &= t <= skip_s + max_s
        ft = ft[keep]
    return ft


def find_csvs(path) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    files = [f for f in p.rglob("*") if f.suffix in (".csv", ".txt") and not f.name.endswith("_summary.csv")]
    return sorted(files)


# --------------------------------------------------------------------------- metrics
def metrics(ft: np.ndarray) -> dict:
    n = len(ft)
    srt = np.sort(ft)
    total_ms = float(ft.sum())
    med = float(np.median(ft))

    def worst_mean(frac: float) -> float:  # mean of the slowest `frac` of frames
        k = max(1, int(math.ceil(n * frac)))
        return float(srt[-k:].mean())

    p99 = float(np.percentile(ft, 99))
    return {
        "frames": n,
        "duration_s": total_ms / 1000.0,
        "avg_fps": 1000.0 * n / total_ms,
        "mean_ms": total_ms / n,
        "median_ms": med,
        "p95_ms": float(np.percentile(ft, 95)),
        "p99_ms": p99,
        "p999_ms": float(np.percentile(ft, 99.9)),
        "max_ms": float(srt[-1]),
        "std_ms": float(ft.std(ddof=1)) if n > 1 else 0.0,
        "low1_fps": 1000.0 / worst_mean(0.01),      # avg FPS over the slowest 1% of frames
        "low01_fps": 1000.0 / worst_mean(0.001),    # avg FPS over the slowest 0.1% of frames
        "p99_fps": 1000.0 / p99,                    # alternative "1% low": FPS at the p99 frame time
        "jitter_ms": float(np.abs(np.diff(ft)).mean()) if n > 1 else 0.0,
        "stutters": int((ft > 2.0 * med).sum()),    # frames slower than 2x the median
        "stutters_per_min": float((ft > 2.0 * med).sum() / (total_ms / 60000.0)),
        "over_50ms": int((ft > 50.0).sum()),
    }


def mean_ci(vals) -> tuple[float, float]:
    v = np.asarray(vals, dtype=float)
    n = len(v)
    if n < 2:
        return float(v.mean()), float("nan")
    t = T975[n - 2] if n - 1 <= len(T975) else 1.96
    return float(v.mean()), float(t * v.std(ddof=1) / math.sqrt(n))


def collect(path, skip, max_s) -> list[dict]:
    rows = []
    for f in find_csvs(path):
        ft = load_frametimes(f, skip, max_s)
        if len(ft) < 100:
            print(f"warning: {f} has only {len(ft)} usable frames; skipped", file=sys.stderr)
            continue
        m = metrics(ft)
        m["file"] = str(f)
        rows.append(m)
    return rows


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*r))


# --------------------------------------------------------------------------- commands
def cmd_summarize(args) -> None:
    all_rows = []
    for p in args.paths:
        all_rows += collect(p, args.skip, args.max_seconds)
    if not all_rows:
        sys.exit("no usable frame-time files found")
    if args.json:
        print(json.dumps(all_rows, indent=2))
        return
    headers = ["file", "frames"] + [m[1] for m in METRICS]
    table = [[Path(r["file"]).parent.name + "/" + Path(r["file"]).name[-24:], str(r["frames"])]
             + [m[2].format(r[m[0]]) for m in METRICS] for r in all_rows]
    print_table(headers, table)


def cmd_compare(args) -> None:
    a = collect(args.a, args.skip, args.max_seconds)
    b = collect(args.b, args.skip, args.max_seconds)
    if not a or not b:
        sys.exit("need at least one usable run in each set")
    print(f"A = {args.a}  ({len(a)} runs)\nB = {args.b}  ({len(b)} runs)\n")
    table = []
    for key, name, fmt, higher_better in METRICS:
        ma, ca = mean_ci([r[key] for r in a])
        mb, cb = mean_ci([r[key] for r in b])
        delta = (mb - ma) / ma * 100.0 if ma else float("nan")
        if math.isnan(ca) or math.isnan(cb):
            verdict = "n/a (need >=2 runs per set)"
        elif abs(ma - mb) > (ca + cb):
            better = (mb > ma) == higher_better
            verdict = "B better" if better else "B worse"
        else:
            verdict = "CIs overlap"
        table.append([name, f"{fmt.format(ma)} +/- {fmt.format(ca) if not math.isnan(ca) else '?'}",
                      f"{fmt.format(mb)} +/- {fmt.format(cb) if not math.isnan(cb) else '?'}",
                      f"{delta:+.1f}%", verdict])
    print_table(["metric", "A mean +/- 95% CI", "B mean +/- 95% CI", "B vs A", "verdict"], table)
    print("\nCIs are Student-t across runs; non-overlap is a screening heuristic, not a formal test.")


def cmd_plot(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for p in args.paths:
        label = Path(p).name
        series = [load_frametimes(f, args.skip, args.max_seconds) for f in find_csvs(p)]
        series = [s for s in series if len(s) >= 100]
        if not series:
            continue
        first = series[0]
        ax[0].plot(np.cumsum(first) / 1000.0, first, lw=0.6, label=label)
        pooled = np.concatenate(series)
        ax[1].hist(pooled, bins=np.geomspace(max(pooled.min(), 0.1), pooled.max(), 80),
                   histtype="step", density=True, label=label)
        tail = np.logspace(2, -2, 300)                      # % of frames that are slower
        ax[2].plot(tail, np.percentile(pooled, np.clip(100 - tail, 0, 100)), label=label)
    ax[0].set(title="Frame time (first run of each set)", xlabel="time (s)", ylabel="ms")
    ax[1].set(title="Frame-time distribution (pooled)", xlabel="ms", ylabel="density",
              xscale="log", yscale="log")
    ax[2].set(title="Tail latency", xlabel="slowest X% of frames", ylabel="ms", xscale="log")
    ax[2].invert_xaxis()
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


def cmd_synth(args) -> None:
    rng = np.random.default_rng(args.seed)
    n = int(args.seconds * args.fps)
    ft = rng.lognormal(math.log(1000.0 / args.fps), args.noise, n)
    spikes = rng.random(n) < args.stutter_rate
    ft[spikes] *= rng.uniform(2.5, 8.0, spikes.sum())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    elapsed = np.cumsum(ft) * 1e6  # ns
    with out.open("w") as f:       # mimic MangoHud: preamble rows, then the data header
        f.write("os,cpu,gpu,kernel\nsynthetic,synthetic,synthetic,synthetic\n")
        f.write("fps,frametime,elapsed\n")
        for x, e in zip(ft, elapsed):
            f.write(f"{1000.0 / x:.3f},{x:.4f},{int(e)}\n")
    print(f"wrote {out} ({n} frames)")


def sysinfo() -> dict:
    def rd(p):
        try:
            return Path(p).read_text().strip()
        except Exception:
            return None

    cpu = next((l.split(":", 1)[1].strip() for l in (rd("/proc/cpuinfo") or "").splitlines()
                if l.startswith("model name")), None)
    gpu = None
    if shutil.which("vulkaninfo"):
        try:
            out = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=20).stdout
            gpu = [l.strip() for l in out.splitlines() if "deviceName" in l or "driverInfo" in l]
        except Exception:
            pass
    return {
        "kernel": platform.release(),
        "cmdline": rd("/proc/cmdline"),
        "cpu": cpu,
        "governor": rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "sched_ext_state": rd("/sys/kernel/sched_ext/state"),
        "sched_ext_ops": rd("/sys/kernel/sched_ext/root/ops"),
        "vulkan": gpu,
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("PROTON_", "DXVK_", "VKD3D_", "WINE", "MESA_", "RADV_", "STAGING_"))},
    }


def cmd_run(args) -> None:
    cmd = [c for i, c in enumerate(args.command) if not (i == 0 and c == "--")]
    if not cmd:
        sys.exit("no command given; put it after `--`")
    base = Path(args.out) / args.label
    timeout = args.timeout or (args.warmup + args.duration + 30)
    for i in range(1, args.runs + 1):
        rd = base / f"run_{i:02d}"
        rd.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["MANGOHUD"] = "1"
        # Verify option names against your MangoHud version (`man mangohud`).
        env["MANGOHUD_CONFIG"] = ",".join([
            f"output_folder={rd}", f"log_duration={args.duration}",
            f"autostart_log={args.warmup}", "log_interval=0", "no_display"])
        full = (["mangohud"] if shutil.which("mangohud") else []) + cmd
        meta = {"label": args.label, "run": i, "command": full,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "warmup_s": args.warmup, "duration_s": args.duration, **sysinfo()}
        print(f"[{args.label}] run {i}/{args.runs}: {' '.join(full)}")
        p = subprocess.Popen(full, env=env, start_new_session=True)
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            meta["timed_out"] = True
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        meta["exit_code"] = p.returncode
        (rd / "meta.json").write_text(json.dumps(meta, indent=2))
        got = find_csvs(rd)
        if got:
            ft = load_frametimes(got[0])
            if len(ft):
                m = metrics(ft)
                print(f"    {m['frames']} frames | avg {m['avg_fps']:.1f} fps | "
                      f"1% low {m['low1_fps']:.1f} | p99.9 {m['p999_ms']:.2f} ms")
        else:
            print("    warning: no CSV produced (is MangoHud installed and does the game use Vulkan?)")
        if i < args.runs and args.cooldown:
            print(f"    cooling down {args.cooldown}s")
            time.sleep(args.cooldown)


# --------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--skip", type=float, default=0.0, help="drop the first N seconds")
        p.add_argument("--max-seconds", type=float, default=None, help="keep at most N seconds after --skip")

    r = sub.add_parser("run", help="capture N runs under MangoHud")
    r.add_argument("--label", required=True, help="condition name, e.g. proton-eevdf")
    r.add_argument("--out", default="results")
    r.add_argument("--runs", type=int, default=5)
    r.add_argument("--duration", type=int, default=120, help="logged seconds per run")
    r.add_argument("--warmup", type=int, default=20, help="seconds before logging starts")
    r.add_argument("--timeout", type=int, default=None, help="kill the command after N seconds")
    r.add_argument("--cooldown", type=int, default=30, help="seconds between runs (thermals)")
    r.add_argument("command", nargs=argparse.REMAINDER, help="-- <command to run>")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("summarize", help="metrics per run")
    s.add_argument("paths", nargs="+")
    s.add_argument("--json", action="store_true")
    common(s)
    s.set_defaults(fn=cmd_summarize)

    c = sub.add_parser("compare", help="A/B comparison of two run sets")
    c.add_argument("a")
    c.add_argument("b")
    common(c)
    c.set_defaults(fn=cmd_compare)

    pl = sub.add_parser("plot", help="time series, histogram, tail plot")
    pl.add_argument("paths", nargs="+")
    pl.add_argument("--out", default="frametimes.png")
    common(pl)
    pl.set_defaults(fn=cmd_plot)

    sy = sub.add_parser("synth", help="generate a synthetic frame-time CSV")
    sy.add_argument("out")
    sy.add_argument("--fps", type=float, default=120)
    sy.add_argument("--seconds", type=float, default=120)
    sy.add_argument("--noise", type=float, default=0.08, help="lognormal sigma of baseline jitter")
    sy.add_argument("--stutter-rate", type=float, default=0.002)
    sy.add_argument("--seed", type=int, default=0)
    sy.set_defaults(fn=cmd_synth)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
