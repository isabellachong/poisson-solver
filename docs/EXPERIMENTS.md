# Reproduce the experiments without scheduler scripts

Run from the project root in **Bash**, after building as described in
[the README](../README.md). These commands launch applications directly inside
resources you have already obtained. Each CPU configuration requires its stated
number of allocated cores; do not run the 256-rank case in a smaller allocation.
GPU configurations below use one physical node, with one or two visible GPUs.

## Common settings

```bash
export RESULTS_DIR="results/performance"
export OMP_NUM_THREADS=1
PYTHON=python3
ITERATIONS=1000
REPEATS=5
mkdir -p "$RESULTS_DIR/cpu" "$RESULTS_DIR/gpu"
```

Use Python's standard library for collection/validation. To plot, prepare an
environment once and select its interpreter for `PYTHON`:

```bash
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements-analysis.txt
PYTHON="$PWD/.venv-analysis/bin/python"
```

All timed runs use the default `mixed` problem, three warm-up iterations followed
by a reset, and `--check-every 0`. Each parallel configuration tests overlap off
and on. CSVs report maximum-across-ranks solve time, including exchange and
computation, excluding setup and final diagnostics. Repetition improves timing
statistics; the accuracy study uses a separate tolerance-based protocol.

## CPU MPI strong and weak scaling

Run this block once for each `P` in the table, in a matching allocation. This is
not an instruction to launch every configuration simultaneously.

| P / CPU cores | Suggested physical nodes | Ranks per node | 2D process grid | 3D process grid |
| ---: | ---: | ---: | --- | --- |
| 1 | 1 | 1 | 1x1 | 1x1x1 |
| 4 | 1 | 4 | 2x2 | 2x2x1 |
| 8 | 1 | 8 | 2x4 | 2x2x2 |
| 16 | 1 | 16 | 4x4 | 4x2x2 |
| 32 | 1 | 32 | 4x8 | 4x4x2 |
| 64 | 1 | 64 | 8x8 | 4x4x4 |
| 128 | 2 | 64 | 8x16 | 8x4x4 |
| 256 | 4 | 64 | 16x16 | 8x8x4 |

The following launcher options target the Open MPI/HPC-X stack used on Vanda.
For another MPI implementation, replace `CPU_MPI` with its equivalent placement
options. The optional PBS host-file environment variable is supplied by the
allocation; no project submission file is sourced. Without it, configure host
placement through your MPI environment.

```bash
P=256                         # Change to a CPU count from the table.
case "$P" in
  1)   GRID2=1x1;   GRID3=1x1x1 ;;
  4)   GRID2=2x2;   GRID3=2x2x1 ;;
  8)   GRID2=2x4;   GRID3=2x2x2 ;;
  16)  GRID2=4x4;   GRID3=4x2x2 ;;
  32)  GRID2=4x8;   GRID3=4x4x2 ;;
  64)  GRID2=8x8;   GRID3=4x4x4 ;;
  128) GRID2=8x16;  GRID3=8x4x4 ;;
  256) GRID2=16x16; GRID3=8x8x4 ;;
  *) echo "Choose P from the table"; exit 1 ;;
esac
RPN="$P"
if (( P > 64 )); then RPN=64; fi
CPU_MPI=(--map-by "ppr:${RPN}:node" --bind-to none
         --mca coll_hcoll_enable 0 --mca coll_ucc_enable 0)
if [[ -n "${PBS_NODEFILE:-}" ]]; then
  CPU_MPI=(--hostfile "$PBS_NODEFILE" "${CPU_MPI[@]}")
fi
CPU_ARGS=()
for argument in "${CPU_MPI[@]}"; do
  CPU_ARGS+=("--launcher-arg=$argument")
done
CPU_BENCH=("$PYTHON" scripts/benchmark.py
  --exe build-cpu/executables/poisson_mpi --mode mpi-cpu
  --iterations "$ITERATIONS" --repeats "$REPEATS"
  --overlap 0 1 --communication host "${CPU_ARGS[@]}")
CPU_OUT="$RESULTS_DIR/cpu/p$P"
mkdir -p "$CPU_OUT"

"${CPU_BENCH[@]}" --kind strong --dim 2 --process-grids "$GRID2" \
  --nx 4096 --ny 4096 --output "$CPU_OUT/strong-2d.csv"
"${CPU_BENCH[@]}" --kind strong --dim 3 --process-grids "$GRID3" \
  --nx 256 --ny 256 --nz 256 --output "$CPU_OUT/strong-3d.csv"
"${CPU_BENCH[@]}" --kind weak --dim 2 --process-grids "$GRID2" \
  --nx 256 --ny 256 --output "$CPU_OUT/weak-2d.csv"
"${CPU_BENCH[@]}" --kind weak --dim 3 --process-grids "$GRID3" \
  --nx 32 --ny 32 --nz 32 --output "$CPU_OUT/weak-3d.csv"
```

Strong sizes are global. Weak sizes are **local per rank**: `benchmark.py`
multiplies them by the process-grid dimensions. At 256 cores, weak grids are
4096x4096 (2D) and 256x256x128 (3D). Local work stays 256² or 32³ points per rank.
The disabled optional HCOLL/UCC collectives address startup failures observed on
Vanda; the command does not force a point-to-point network transport. Binding is
`none`, matching the validated CPU workflow.

## GPU strong and weak scaling

Use `G=1` in a one-node/one-GPU allocation with eight MPI slots. Repeat with
`G=2` in a one-node/two-GPU allocation with sixteen slots. The experiments
allocated 36 supporting CPU cores for one GPU and 72 for two GPUs. All ranks
must see the intended allocated device list. No CPU-only CUDA fallback is used.

```bash
G=1                           # Change to 2 in the two-GPU allocation.
case "$G" in
  1) GRID2=2x4; GRID3=2x2x2 ;;
  2) GRID2=4x4; GRID3=4x2x2 ;;
  *) echo "Use G=1 or G=2"; exit 1 ;;
esac
RANKS=$((8*G))
GPU_BENCH=("$PYTHON" scripts/benchmark.py
  --exe build-gpu/executables/poisson_mpi --mode mpi-cuda
  --iterations "$ITERATIONS" --repeats "$REPEATS"
  --overlap 0 1 --communication cuda-aware
  --launcher-arg=--map-by "--launcher-arg=ppr:${RANKS}:node"
  --launcher-arg=--bind-to --launcher-arg=none)
GPU_OUT="$RESULTS_DIR/gpu/g$G"
mkdir -p "$GPU_OUT"

"${GPU_BENCH[@]}" --kind strong --dim 2 --process-grids "$GRID2" \
  --nx 4096 --ny 4096 --output "$GPU_OUT/strong-2d.csv"
"${GPU_BENCH[@]}" --kind strong --dim 3 --process-grids "$GRID3" \
  --nx 256 --ny 256 --nz 256 --output "$GPU_OUT/strong-3d.csv"
"${GPU_BENCH[@]}" --kind weak --dim 2 --process-grids "$GRID2" \
  --nx 1024 --ny 1024 --output "$GPU_OUT/weak-2d.csv"
"${GPU_BENCH[@]}" --kind weak --dim 3 --process-grids "$GRID3" \
  --nx 128 --ny 128 --nz 128 --output "$GPU_OUT/weak-3d.csv"
```

Verify `gpus_used=G`, `mpi_ranks=8*G`, and `max_ranks_per_gpu=8` in every CSV.
GPU update/packing block size remains the default 256 threads. Strong global
sizes are fixed. Weak sizes grow as follows, keeping work/GPU and ranks/GPU fixed:

| GPUs | Ranks | Global weak 2D grid | Global weak 3D grid |
| ---: | ---: | --- | --- |
| 1 | 8 | 2048x4096 | 256x256x256 |
| 2 | 16 | 4096x4096 | 512x256x256 |

## Fixed-one-GPU sharing

Run only in a one-node/one-GPU allocation with eight MPI slots. Hardware and
global problem stay fixed while active ranks increase from 1 to 4 to 8.

```bash
SHARING_BENCH=("$PYTHON" scripts/benchmark.py
  --exe build-gpu/executables/poisson_mpi --mode mpi-cuda --kind sharing
  --iterations "$ITERATIONS" --repeats "$REPEATS"
  --overlap 0 1 --communication cuda-aware
  --launcher-arg=--map-by --launcher-arg=ppr:8:node
  --launcher-arg=--bind-to --launcher-arg=none)
mkdir -p "$RESULTS_DIR/gpu/sharing"
"${SHARING_BENCH[@]}" --dim 2 --process-grids 1x1 2x2 2x4 \
  --nx 4096 --ny 4096 --output "$RESULTS_DIR/gpu/sharing/sharing-2d.csv"
"${SHARING_BENCH[@]}" --dim 3 --process-grids 1x1x1 2x2x1 2x2x2 \
  --nx 256 --ny 256 --nz 256 --output "$RESULTS_DIR/gpu/sharing/sharing-3d.csv"
```

Check `gpus_used=1`. No MPS service is started by the code. This is a rank-sharing
study, not an increase in GPU hardware.

## Serial runtime versus problem size

Use one CPU core and the same CPU build as the MPI CPU measurements:

```bash
mkdir -p "$RESULTS_DIR/cpu/serial"
"$PYTHON" scripts/benchmark.py --exe build-cpu/executables/poisson_serial \
  --mode serial --kind serial --dim 2 --sizes 64 128 256 512 1024 2048 4096 \
  --iterations "$ITERATIONS" --repeats "$REPEATS" \
  --output "$RESULTS_DIR/cpu/serial/serial-2d.csv"
"$PYTHON" scripts/benchmark.py --exe build-cpu/executables/poisson_serial \
  --mode serial --kind serial --dim 3 --sizes 16 32 64 128 256 \
  --iterations "$ITERATIONS" --repeats "$REPEATS" \
  --output "$RESULTS_DIR/cpu/serial/serial-3d.csv"
```

## Plot collected measurements

Plotting can run on the laptop or a suitable compute allocation. Set
`RESULTS_DIR` to the study you actually collected or downloaded. This discovery
also accepts historical subfolder names; only benchmark CSVs are selected.
Do not combine independent studies, duplicate configurations or iteration counts.

```bash
shopt -s nullglob
cpu_inputs=("$RESULTS_DIR"/cpu/*/strong-*.csv
            "$RESULTS_DIR"/cpu/*/weak-*.csv
            "$RESULTS_DIR"/cpu/*/serial-*.csv)
gpu_inputs=("$RESULTS_DIR"/gpu/*/strong-*.csv
            "$RESULTS_DIR"/gpu/*/weak-*.csv
            "$RESULTS_DIR"/gpu/*/sharing-*.csv)
if (( ${#cpu_inputs[@]} > 0 )); then
  "$PYTHON" scripts/plot_scaling.py --input "${cpu_inputs[@]}" \
    --output-dir "$RESULTS_DIR/cpu/figures"
fi
if (( ${#gpu_inputs[@]} > 0 )); then
  "$PYTHON" scripts/plot_scaling.py --input "${gpu_inputs[@]}" \
    --output-dir "$RESULTS_DIR/gpu/figures"
fi
```

For the dedicated two-panel serial figure, select the two serial CSVs:

```bash
"$PYTHON" scripts/plot_serial.py \
  --input "$RESULTS_DIR/cpu/serial/serial-2d.csv" \
          "$RESULTS_DIR/cpu/serial/serial-3d.csv" \
  --output-dir "$RESULTS_DIR/cpu/serial/figures"
```

Figures use medians with min/max runtime bars. Time, problem-size and resource
axes are logarithmic; speedup and efficiency values are linear. Outputs are PNG
and summary CSV; the dedicated serial plot also writes input metadata JSON.
No SVG or caption files are generated.

Strong speedup is `T(P0)/T(P)` and efficiency is speedup divided by `P/P0`.
Weak efficiency is `Tweak(P0)/Tweak(P)`. Baselines are the smallest measured
resource count in each dimension/overlap series: normally one CPU MPI rank or
one GPU with eight ranks. Sharing uses the one-rank/one-GPU reference and has no
GPU hardware-efficiency curve. Serial-to-MPI speedup requires a separately
matched serial reference; the plotted MPI baseline is not that reference.

Parallel plots show runtime, speedup and efficiency against resource count.
Weak-scaling runtime-versus-unknowns/nonzeros figures are excluded; rerunning
the plotter removes those obsolete PNGs for the weak series being plotted.
Serial runtime-versus-unknowns/nonzeros figures are retained for the serial size
study. A separate parallel size sweep at fixed core/GPU count was not measured.

## Compare serial, MPI CPU and MPI+CUDA

Use the serial size-sweep CSVs and **strong-scaling** CSVs from the same study.
The comparison selects matching global grids (4096² and 256³ by default),
requires five distinct repeats, and checks problem, iteration count and warm-up
consistency. It rejects weak/sharing inputs. It does not run any solver.

```bash
comparison_inputs=("$RESULTS_DIR"/cpu/*/serial-*.csv
                   "$RESULTS_DIR"/cpu/*/strong-*.csv
                   "$RESULTS_DIR"/gpu/*/strong-*.csv)
"$PYTHON" scripts/plot_comparison.py --input "${comparison_inputs[@]}" \
  --iterations 1000 --repeats 5 --size-2d 4096 --size-3d 256 \
  --output-dir "$RESULTS_DIR/comparison"
```

Run in Bash with `shopt -s nullglob` as above. Inputs may use the historical
measurement-folder names or the direct-command layout; change `RESULTS_DIR`
to the directory actually containing your downloaded study. Keep compiler,
hardware/runtime environment and timing scope matched when selecting inputs;
CSV validation cannot independently establish compiler or device provenance.

Outputs are:

```text
comparison/comparison_runtime_overlap0.png
comparison/comparison_runtime_overlap1.png
comparison/comparison_speedup_overlap0.png
comparison/comparison_speedup_overlap1.png
comparison/comparison_summary.csv
```

Each figure has 2D and 3D panels. Runtime uses a logarithmic time axis and
median/min-max bars. Speedup uses a linear axis and ratios of median times,
without error bars. The horizontal axis is categorical execution configuration,
so it is not logarithmic. Serial is the same baseline in both overlap figures;
overlap has no meaning for that executable. Only measured CPU/GPU configurations
are plotted; a missing 256-core run is not filled in or extrapolated.

Acceleration is `serial_median_seconds / parallel_median_seconds`. The CSV also
reports CPU efficiency relative to serial as acceleration divided by CPU rank
count, assuming one allocated core per CPU rank. GPU acceleration is not GPU
scaling efficiency: one GPU plus its supporting CPU allocation is a different
hardware configuration from one CPU core. Keep GPU scaling efficiency relative
to the one-GPU MPI+CUDA baseline in the separate scaling figures.
