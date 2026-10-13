# ftharness: frame-time capture & analysis

One file (`ft.py`), numpy required, matplotlib for plots. Tested end to end on synthetic data; the MangoHud capture path still needs a first run on your machine.

## Setup
```bash
sudo apt install mangohud        # or your distro's package
pip install numpy matplotlib
```

## Workflow
```bash
# 0. sanity-check the pipeline without a game
python3 ft.py synth results/test/run_01/a.csv

# 1. capture: 5 runs per condition, same scene every time
python3 ft.py run --label proton-stock --runs 5 --warmup 20 --duration 120 -- <your benchmark command>

# Steam titles: set the launch option instead, then point summarize at the folder
#   MANGOHUD_CONFIG=output_folder=/path/results/proton-stock/run_01,autostart_log=20,log_duration=120,log_interval=0,no_display mangohud %command%

# 2. inspect
python3 ft.py summarize results/proton-stock
python3 ft.py compare results/proton-stock results/proton-tuned
python3 ft.py plot results/proton-stock results/proton-tuned --out tails.png
```
Any single-column file of frame times in ms also works, so you can feed in captures from other tools.

## Metric definitions
- **avg FPS**: frames / total time (not the mean of per-frame FPS).
- **1% / 0.1% low FPS**: 1000 / mean frame time of the slowest 1% / 0.1% of frames. `p99_fps` (FPS at the p99 frame time) is also in `--json` output, since tools disagree on the definition. State yours in papers.
- **p99 / p99.9 ms**: tail frame times. **jitter**: mean absolute frame-to-frame delta.
- **stutters**: frames slower than 2x the run's median frame time.

## Methodology notes
- Interleave conditions (A, B, A, B...) rather than all A then all B, to avoid thermal and cache drift.
- Pin the CPU governor, close background load, and record the same in-game scene or built-in benchmark every run.
- Use at least 5 runs per condition; with 2-3 runs the confidence intervals are very wide.
- `meta.json` stores kernel, governor, `sched_ext` state, Vulkan device/driver, and Proton/DXVK env vars for each run.
- Verify the MangoHud option names (`output_folder`, `log_duration`, `autostart_log`, `log_interval`) against `man mangohud` for your version.

## Next experiments this feeds
Frame-time deltas become the dependent variable for: scheduler policy comparison (EEVDF vs `uclamp`/cpuset vs a `sched_ext` policy), NTSYNC/fsync on vs off, and container vs bare metal. Add eBPF run-queue latency traces alongside each run to explain *why* a delta appears.
