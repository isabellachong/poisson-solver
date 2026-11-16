# Jacobi-Poisson in 2D and 3D

Double-precision C++17 Jacobi solvers for `-laplacian(u)=f` on the unit square or cube, with Dirichlet boundaries and five-/seven-point finite differences.

| Executable | Backend |
| --- | --- |
| `poisson_serial` | One CPU process, no MPI initialization |
| `poisson_mpi` in a CPU build | One CPU thread per MPI rank |
| `poisson_mpi` in a CUDA build | MPI decomposition with CUDA computation per rank |

This README is the complete execution and reproduction guide. All commands run from the project root in **Bash** unless explicitly labelled laptop commands. Obtain the stated CPU/GPU resources before executing solvers on a cluster.

## Contents

1. [Requirements and build](#requirements-and-build)
2. [Resource allocation and shell setup](#resource-allocation-and-shell-setup)
3. [Common settings and plotting environment](#common-settings-and-plotting-environment)
4. [Run the solvers and configure parameters](#run-the-solvers-and-configure-parameters)
5. [Algorithm and parallel design](#algorithm-and-parallel-design)
6. [Correctness validation](#correctness-validation)
7. [Main performance study](#main-performance-study)
8. [Spatial accuracy study](#spatial-accuracy-study)
9. [Plots and performance analysis](#plots-and-performance-analysis)
10. [Supplementary layout and transport studies](#supplementary-layout-and-transport-studies)
11. [Results organization and transfer](#results-organization-and-transfer)
12. [Source organization](#source-organization)

## Requirements and build

Use CMake >=3.20, a C++17 compiler, Python 3 and MPI development tools. CUDA builds also require NVCC, a compatible NVIDIA driver/toolkit and allocated GPUs. CUDA-aware transfers require CUDA-aware MPI. Use matching compiler/MPI installations for build and execution, with fresh build directories when changing stacks. Build inside an allocation with at least four CPU cores; GPU architecture detection additionally needs a visible allocated GPU.

The Vanda measurements used the following module stack. Initialize it in each compute shell when available on your installation:

```bash
if ! type module >/dev/null 2>&1; then source /app1/ebenv; fi
module purge
module load nvhpc-hpcx-cuda12/26.1
export OMP_NUM_THREADS=1
```

CPU build, including serial and MPI:

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DPOISSON_ENABLE_MPI=ON -DPOISSON_ENABLE_CUDA=OFF \
  -DCMAKE_CXX_COMPILER=mpicxx -DMPI_CXX_COMPILER=mpicxx
cmake --build build-cpu --parallel 4
```

CUDA build:

```bash
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
  -DPOISSON_ENABLE_MPI=ON -DPOISSON_ENABLE_CUDA=ON \
  -DMPI_CXX_COMPILER=mpicxx
cmake --build build-gpu --parallel 4
```

If needed, set `-DCMAKE_CUDA_ARCHITECTURES=NN` to the GPU's compute capability; `native` requires CMake >=3.24 and a visible GPU. For serial only, disable both `POISSON_ENABLE_MPI` and `POISSON_ENABLE_CUDA`. If CMake is unavailable:

```bash
python3 -m venv .venv-build
.venv-build/bin/python -m pip install 'cmake>=3.24,<4'
export PATH="$PWD/.venv-build/bin:$PATH"
```

Run `cmake -S ... -B ...` again after changing test registration or build options. Use a fresh build directory when changing compiler/MPI stacks. Existing Vanda executables are Linux binaries; rebuild from source on another platform rather than transferring build directories to a laptop.

## Resource allocation and shell setup

On Vanda, request an allocation from the login shell. Each interactive request below opens a compute shell; run its work and then `exit` to release it. Choose one request appropriate to the task, rather than running every request at once. The commands use the account/queue selected by the site's defaults.

```bash
# CPU build and extended validation: eight cores on one host.
qsub -I -l select=1:ncpus=8:mpiprocs=8:ompthreads=1:mem=16gb -l walltime=01:00:00

# Serial timings or spatial accuracy: one core.
qsub -I -l select=1:ncpus=1:mpiprocs=1:ompthreads=1:mem=4gb -l walltime=01:00:00

# One GPU, accessible to up to eight MPI ranks on one host.
qsub -I -l select=1:ncpus=36:mpiprocs=8:ompthreads=1:ngpus=1 -l walltime=01:00:00

# Two GPU-containing chunks packed onto one physical host: sixteen MPI slots.
qsub -I -l select=2:ncpus=36:mpiprocs=8:ompthreads=1:ngpus=1 \
  -l place=pack -l walltime=01:00:00
```

Interactive resource limits can be smaller than batch limits. In particular, **do not request 256 cores through the limited interactive queue**. Run that case in a suitable batch allocation obtained through the site's scheduler. Its resource specification is `select=4:ncpus=64:mpiprocs=64:ompthreads=1` with `place=scatter`; the program commands below execute within that allocation. A project-specific submission file is not part of the required source interface.

For CPU counts up to 64, the main study used one host with one core per rank; 128 used two hosts with 64 ranks each and 256 used four. GPU hardware scaling used eight ranks per GPU on one physical host. The scheduler may adjust memory and CPU allocations according to site policy. Verify actual hosts/slots with `sort "$PBS_NODEFILE" | uniq -c` when that variable is available, and check the solver's actual GPU counts and UUID mapping.

In each new compute shell, enter the shared project directory and initialize the module environment from the build section. Adjust this variable to your remote installation:

```bash
PROJECT_DIR="$HOME/poisson-solver"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=1
```

Shell variables/arrays do not persist between allocations. Reapply the common settings and the relevant launcher arguments each time. Build before serial runs, and avoid concurrent rebuilds of the same shared build directory.

## Common settings and plotting environment

```bash
export RESULTS_DIR="results/performance"
export OMP_NUM_THREADS=1
PYTHON=python3
ITERATIONS=1000
REPEATS=5
mkdir -p "$RESULTS_DIR/cpu" "$RESULTS_DIR/gpu"
```

Collection and executable validation use the Python standard library. Plotting and the analysis regression tests require Matplotlib. Prepare an environment once and select its interpreter for `PYTHON`:

```bash
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements-analysis.txt
PYTHON="$PWD/.venv-analysis/bin/python"
```

Timed runs use the mixed problem, three warm-up steps followed by a reset, `check_every=0`, and both parallel overlap settings. The explicit 1,000 iterations and five repeats override benchmark defaults of 500 and three. Block size is 256.

The commands in this README specify the report protocol explicitly. Older `jobs/` wrappers have different defaults: three repeats and, for GPU strong scaling and sharing, an 8192² grid. If reusing those wrappers, override `POISSON_ITERATIONS=1000`, `POISSON_REPEATS=5`, `POISSON_STRONG_2D=4096` and `POISSON_CPU_STRONG_2D=4096`. The README commands do not require those wrappers.

`solve_seconds` is the maximum rank time for iterations, exchange, packing and computation. Setup includes communicator/device mapping, arrays and warm-up; diagnostics include final residual/error evaluation and optional solution writing. MPI initialization/finalization, the start barrier and cleanup are excluded. These are solver phase times, not end-to-end wall time.

Choose a fresh `RESULTS_DIR` per study, reusing it across allocations. Existing CSVs are rejected; failed runs can leave partial files. Check completeness and rerun failures under fresh output paths. Record compiler/MPI/CUDA versions, hardware, placement, binding and device visibility alongside the measurements.

## Run the solvers and configure parameters

```bash
# Serial 2D and 3D: 1,000 fixed iterations.
./build-cpu/executables/poisson_serial \
  --dim 2 --nx 128 --ny 128 --iterations 1000 --check-every 0
./build-cpu/executables/poisson_serial \
  --dim 3 --nx 32 --ny 32 --nz 32 --iterations 1000 --check-every 0

# Four MPI ranks on four allocated CPU cores.
mpiexec -n 4 ./build-cpu/executables/poisson_mpi \
  --dim 2 --nx 128 --ny 128 --px 2 --py 2 \
  --iterations 1000 --check-every 0 --communication host --overlap 1

# Eight MPI ranks sharing one allocated GPU on one node.
mpiexec -n 8 ./build-gpu/executables/poisson_mpi \
  --dim 3 --nx 64 --ny 64 --nz 64 --px 2 --py 2 --pz 2 \
  --iterations 1000 --check-every 0 --communication cuda-aware --overlap 1

# Sixteen MPI ranks sharing two allocated GPUs on one host.
mpiexec -n 16 ./build-gpu/executables/poisson_mpi \
  --dim 3 --nx 64 --ny 64 --nz 64 --px 4 --py 2 --pz 2 \
  --iterations 1000 --check-every 0 --communication cuda-aware --overlap 1
```

| Setting | Meaning |
| --- | --- |
| `--dim`, `--nx`, `--ny`, `--nz` | Dimension and global interior point counts; 2D ignores `nz` |
| `--px`, `--py`, `--pz` | Process grid; zero selects an unspecified axis automatically |
| `--iterations K --check-every 0` | Exactly K measured iterations |
| `--check-every Q --rtol R --atol A` | Check residual every Q steps and stop at tolerance |
| `--overlap 0` or `1` | Interior update after/before MPI completion |
| `--warmup K` | Untimed iterations followed by an initial-state reset |
| `--communication host` or `cuda-aware` | CUDA boundary-data transport; host staging still computes on GPU |
| `--block-size N` | CUDA update/packing threads per block; default 256 |
| `--config namelist.txt` | Explicitly load configuration; CLI overrides file settings |

Run either executable with `--help` for all options. Defaults are 2D, 64 points per axis, 500 iterations, three warm-up steps and overlap off. Distributed grids must split at least two axes, multiply to rank count and give each rank owned points. One-rank baselines are supported; two-rank decompositions are rejected.

GPU selection occurs before MPI initialization: node-local rank modulo visible GPU count. Ranks must see the intended allocated devices. Check `gpus_used`, `max_ranks_per_gpu` and the reported host/device UUID mapping. No MPS service is started. Optional `--solution-prefix results/u` writes owned values per rank; create its parent directory first.

## Algorithm and parallel design

Interior values start at zero. Each Jacobi step uses the previous iterate's neighbours and the analytic forcing, with spacing `h[a]=1/(N[a]+1)` and fixed physical boundaries. MPI exchanges packed faces into neighbouring-value storage; CUDA threads update local points. With overlap enabled, strictly local interior work precedes MPI completion, then boundary work follows. CUDA events order received data before boundary updates. Actual overlap depends on MPI progress, hardware and GPU sharing; stencil tiling is not implemented.

The matrix is implicit. Its dimension is the number of interior unknowns M: `M=Nx*Ny`, `nnz=5*M-2*(Nx+Ny)` in 2D; `M=Nx*Ny*Nz`, `nnz=7*M-2*(Nx*Ny+Nx*Nz+Ny*Nz)` in 3D.

| Problem | Exact solution used to generate forcing and boundaries |
| --- | --- |
| `sine` | Product of `sin(pi*x[a])` over active dimensions |
| `mixed` (default) | Sine solution plus `0.25*sin(2*pi*x)*sin(3*pi*y)`, also multiplied by `sin(pi*z)` in 3D |
| `polynomial` | `x+2*y` (+`3*z` in 3D) plus the product of `x[a]*(1-x[a])` |
| `zero` | Zero |

Forcing is the analytic negative Laplacian. Convergence requires `norm_inf(f-A_h*u) <= atol + rtol*norm_inf(f)`; the defaults are `rtol=1e-8` and `atol=1e-10`. Residual evaluation refreshes neighbouring values and reduces a host scalar with MPI; CUDA uses a local two-stage maximum reduction. Exit codes are 0 for completed fixed iterations or convergence, 2 for a tolerance solve reaching its limit, and 1 for an error.

Each iteration reads the old solution and writes a separate new solution. For spacing `h[a]=1/(N[a]+1)`, the update is:

```text
next[i] = (f[i] + sum_a (old[i-e_a] + old[i+e_a])/h[a]^2)
          / (2 * sum_a 1/h[a]^2)
```

Only face neighbours are required. Noncontiguous outgoing faces are packed into contiguous buffers, and incoming values are unpacked before boundary updates. In CUDA-aware mode MPI communicates device buffers; host staging uses reusable pinned host buffers. Packing completes before MPI sends access those buffers. With overlap enabled, the interior kernel may run while MPI progresses; a CUDA event orders received data before boundary work. The compute stream completes before swapping solution arrays. No global barrier is inserted per iteration.

## Correctness validation

### Local CPU checks

```bash
ctest --test-dir build-cpu --output-on-failure
python3 tests/validate_mpi.py --serial-only \
  --serial build-cpu/executables/poisson_serial
POISSON_SERIAL="$PWD/build-cpu/executables/poisson_serial" \
  .venv-analysis/bin/python -m unittest discover -s tests \
  -p test_analysis_scripts.py -v
```

The eight CTest groups cover decomposition and sparse counts, stencil/boundary correctness, update-region coverage, in-process face exchange in 2D and 3D, convergence/spatial order, configuration rejection, and the additional `cartesian_part3` oriented/uneven/thin partitions. They use CPU arrays in one process and do not exercise actual MPI communication or CUDA kernels.

The serial validator independently computes nine polynomial-problem updates on 9x7 and 9x7x5 grids. It checks every owned value, unique/complete ownership, residuals and reset after two warm-up iterations. Two further checks test initial convergence and nonconvergence exit code 2: four executable checks in total. The consolidated regression file contains seven main collection/analysis checks, seven supplementary layout/transport checks and four parser checks: 18 total. They cover provenance, overwrite protection, matched serial baselines, separation of settings/layouts, pairing and malformed summaries. Install plotting dependencies before running them; synthetic fixtures remain in temporary test directories.

### MPI and CUDA checks

With eight allocated CPU cores:

```bash
python3 tests/validate_mpi.py \
  --serial build-cpu/executables/poisson_serial \
  --parallel build-cpu/executables/poisson_mpi --backend cpu
```

With eight MPI slots and one allocated GPU visible to all ranks:

```bash
python3 tests/validate_mpi.py \
  --serial build-gpu/executables/poisson_serial \
  --parallel build-gpu/executables/poisson_mpi --backend cuda \
  --communication both
```

Default process grids are 2x2 and 2x2x2. Both schedules are checked, and CUDA validation covers host-staged and CUDA-aware transfers. Including serial checks, the CPU suite has eight executable checks and the GPU suite twelve.

Use repeated `--launcher-arg=VALUE` arguments for placement. For Open MPI/HPC-X, append `--launcher-arg=--map-by --launcher-arg=ppr:8:node --launcher-arg=--bind-to --launcher-arg=none` to the command. CPU runs can also append `--launcher-arg=--mca --launcher-arg=coll_hcoll_enable --launcher-arg=0 --launcher-arg=--mca --launcher-arg=coll_ucc_enable --launcher-arg=0`. For multi-node PBS allocations, add `--launcher-arg=--hostfile "--launcher-arg=$PBS_NODEFILE"` and adjust ranks per node. Other MPI stacks require equivalent placement options.

Validate each intended timing layout first. The experiment commands do this using `--points 33 35 17` for large CPU process grids; 256 ranks use `--grid2d 16x16 --grid3d 8x8x4`. To check inter-node execution cheaply, allocate two nodes with two cores each and use `--grid2d 2x2 --grid3d 2x2x1`, mapping two ranks per node. Check actual hosts and GPU UUID assignments. Temporary solution files use the shared project directory and are removed automatically; every node must be able to access it. Each validator solve has a 180-second timeout.

### Extended Cartesian executable suite

Append `--suite part3` to the serial or MPI validator commands above. It adds 2x3/3x2 orientations, thin owned regions, uneven partitions, 3D splits in different axis pairs and explicit in-loop residual/convergence checks. `--work-dir PATH` chooses shared storage for temporary per-rank solution files.

```bash
python3 tests/validate_mpi.py --serial-only --suite part3 \
  --serial build-cpu/executables/poisson_serial
```

This performs 12 serial checks. With enough slots for eight ranks, the extended CPU suite performs 36 executable checks. Each separately invoked single-transport CUDA suite also performs 36 checks; running host and CUDA-aware separately gives 72 total, including repeated serial controls. Add the launcher arguments from the MPI validation section as needed. These are test-suite counts, not performance samples.

```bash
for transport in host cuda-aware; do
  python3 tests/validate_mpi.py --suite part3 --backend cuda \
    --communication "$transport" \
    --serial build-gpu/executables/poisson_serial \
    --parallel build-gpu/executables/poisson_mpi
 done
```

The supplementary `layout_study.py audit` rejects missing, duplicate or extra matrix entries and inconsistent numerical diagnostics. The plot and transport summaries require paired repeats within identical allocation/hardware/settings. The accepted supplementary study has 180 CPU and 160 GPU records; the incomplete 29-record 128³ GPU sweep must remain excluded from those formal summaries.

Supplementary collection, binding, analysis and audits use `layout_study.py`. The three Python regression suites are in `test_analysis_scripts.py`; source code is provided in the repository rather than a report appendix.

## Main performance study

The main study uses the mixed problem, 1,000 measured iterations, three warm-up iterations followed by a reset, and five repeats per configuration. The supplementary 2,000-iteration studies later in this README are separate.

| Study | Resource/workload choices | Complete raw record count |
| --- | --- | ---: |
| CPU strong and weak scaling | 1/4/8/16/32/64/128/256 cores, 2D/3D, overlap off/on | 320 |
| GPU strong and weak scaling | 1/2 GPUs, eight ranks/GPU, 2D/3D, overlap off/on | 80 |
| GPU sharing | One GPU with 1/4/8 ranks, 2D/3D, overlap off/on | 60 |
| Serial size sweep | Seven 2D sizes and five 3D sizes | 60 |

The complete study contains 520 performance records. Each repeat is a separate executable invocation within an allocation. Five repeats are not five independent allocations. Record CPU/GPU models, compiler/MPI/CUDA versions, host placement, binding and MPS status; the CSV cannot independently establish those conditions.

### CPU MPI strong and weak scaling

Run this block once per tabled `P`, inside its matching allocation.

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

Launcher arguments target Open MPI/HPC-X. Other MPI implementations need equivalent placement options. `PBS_NODEFILE` comes from the allocation; without it, configure host placement through MPI.

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
# Check this actual process layout before recording timings.
"$PYTHON" tests/validate_mpi.py \
  --serial build-cpu/executables/poisson_serial \
  --parallel build-cpu/executables/poisson_mpi --backend cpu \
  --points 33 35 17 --grid2d "$GRID2" --grid3d "$GRID3" "${CPU_ARGS[@]}"

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

Strong sizes are global. Weak sizes are **local per rank**: `benchmark.py` multiplies them by the process-grid dimensions. At 256 cores, weak grids are 4096x4096 (2D) and 256x256x128 (3D). Local work stays 256² or 32³ points per rank. Optional HCOLL/UCC collectives are disabled to avoid observed Vanda startup failures; point-to-point transport is unchanged. CPU binding is `none`.

### GPU strong and weak scaling

Use `G=1` with one GPU/eight MPI slots/36 CPU cores on one node; repeat with `G=2` and two GPUs/sixteen slots/72 cores on one node. All ranks must see the intended allocated devices.

```bash
G=1                           # Change to 2 in the two-GPU allocation.
case "$G" in
  1) GRID2=2x4; GRID3=2x2x2 ;;
  2) GRID2=4x4; GRID3=4x2x2 ;;
  *) echo "Use G=1 or G=2"; exit 1 ;;
esac
RANKS=$((8*G))
GPU_ARGS=(--launcher-arg=--map-by "--launcher-arg=ppr:${RANKS}:node"
          --launcher-arg=--bind-to --launcher-arg=none)
if [[ -n "${PBS_NODEFILE:-}" ]]; then
  GPU_ARGS+=(--launcher-arg=--hostfile "--launcher-arg=$PBS_NODEFILE")
fi
"$PYTHON" tests/validate_mpi.py \
  --serial build-gpu/executables/poisson_serial \
  --parallel build-gpu/executables/poisson_mpi --backend cuda \
  --grid2d "$GRID2" --grid3d "$GRID3" --communication both "${GPU_ARGS[@]}"
GPU_BENCH=("$PYTHON" scripts/benchmark.py
  --exe build-gpu/executables/poisson_mpi --mode mpi-cuda
  --iterations "$ITERATIONS" --repeats "$REPEATS"
  --overlap 0 1 --communication cuda-aware
  "${GPU_ARGS[@]}")
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

Verify `gpus_used=G`, `mpi_ranks=8*G`, and `max_ranks_per_gpu=8` in every CSV. GPU update/packing block size remains the default 256 threads. Strong global sizes are fixed. Weak sizes grow as follows, keeping work/GPU and ranks/GPU fixed:

| GPUs | Ranks | Global weak 2D grid | Global weak 3D grid |
| ---: | ---: | --- | --- |
| 1 | 8 | 2048x4096 | 256x256x256 |
| 2 | 16 | 4096x4096 | 512x256x256 |

### Fixed-one-GPU sharing

Run only in a one-node/one-GPU allocation with eight MPI slots. Hardware and global problem stay fixed while active ranks increase from 1 to 4 to 8.

```bash
SHARING_ARGS=(--launcher-arg=--map-by --launcher-arg=ppr:8:node
              --launcher-arg=--bind-to --launcher-arg=none)
if [[ -n "${PBS_NODEFILE:-}" ]]; then
  SHARING_ARGS+=(--launcher-arg=--hostfile "--launcher-arg=$PBS_NODEFILE")
fi
"$PYTHON" tests/validate_mpi.py \
  --serial build-gpu/executables/poisson_serial \
  --parallel build-gpu/executables/poisson_mpi --backend cuda \
  --communication both "${SHARING_ARGS[@]}"
SHARING_BENCH=("$PYTHON" scripts/benchmark.py
  --exe build-gpu/executables/poisson_mpi --mode mpi-cuda --kind sharing
  --iterations "$ITERATIONS" --repeats "$REPEATS"
  --overlap 0 1 --communication cuda-aware
  "${SHARING_ARGS[@]}")
mkdir -p "$RESULTS_DIR/gpu/sharing"
"${SHARING_BENCH[@]}" --dim 2 --process-grids 1x1 2x2 2x4 \
  --nx 4096 --ny 4096 --output "$RESULTS_DIR/gpu/sharing/sharing-2d.csv"
"${SHARING_BENCH[@]}" --dim 3 --process-grids 1x1x1 2x2x1 2x2x2 \
  --nx 256 --ny 256 --nz 256 --output "$RESULTS_DIR/gpu/sharing/sharing-3d.csv"
```

Check `gpus_used=1`. No MPS service is started by the code. This is a rank-sharing study, not an increase in GPU hardware.

### Serial runtime versus problem size

Use one CPU core and the same CPU build as the MPI CPU measurements:

```bash
"$PYTHON" tests/validate_mpi.py --serial-only \
  --serial build-cpu/executables/poisson_serial
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

### Measurement checks and limitations

Before plotting, check five distinct repeats (0–4), 1,000 iterations, three warm-up steps, `check_every=0`, the mixed problem, intended grids/resources and both schedules for parallel runs. A complete study has 320 CPU scaling rows, 80 GPU scaling rows, 60 sharing rows and 60 serial rows. Failed benchmarks can leave partial CSVs. Record compiler/runtime versions, hardware, placement and GPU visibility; CSVs do not independently establish that provenance.

The scaling plotter checks constant global grid shape for strong scaling, constant unknowns per resource for weak scaling, consistent process-grid shapes, and separate label/warm-up/device/block-size series. `--serial-reference` matches serial grid/problem/iterations/warm-up/label before computing acceleration. `--serial-only` rejects parallel inputs and records serial input hashes. The common plotter does not enforce a complete repeat matrix; check coverage yourself or use the supplementary matrix audit for layout studies. The comparison plotter also requires the requested repeat count and matched grid/transport/block settings; it does not require every planned resource count. Check coverage through 256 CPU cores yourself. Recompute a selected median/min/max and the speedup/ efficiency ratios defined in [plots and analysis](#plots-and-performance-analysis).

Report only checks actually completed. CPU tests do not establish GPU or inter-node correctness. Numerical agreement does not establish GPU memory safety or achieved overlap; those need sanitizer/timeline checks on the cluster.

## Spatial accuracy study

Use one CPU core and the plotting environment from the common-settings section:

```bash
.venv-analysis/bin/python scripts/accuracy_study.py \
  --exe build-cpu/executables/poisson_serial \
  --output-dir results/accuracy/study --plot
```

The output directory must be new. Six sine solves use N=7/15/31 interior points per axis in both dimensions, `rtol=1e-11`, `atol=1e-12`, two warm-up steps, checks every ten iterations and a 50,000-iteration limit. An independent finite-step discrete sine-eigenmode reference checks owned values and maximum/ root-mean-square errors. Residuals must meet tolerance and successive spatial orders must lie strictly between 1.9 and 2.1.

Outputs are `refinement.csv`, `refinement.png`, `run.json` and per-rank solution CSVs. This converged accuracy study is separate from fixed-iteration timing.

## Plots and performance analysis

### Scaling and serial figures

Set `RESULTS_DIR` to the completed study and `PYTHON` to the plotting interpreter. Apply the [measurement checks](#measurement-checks-and-limitations) first; do not combine independent studies or duplicate repeats. Globs accept any subfolder names one level below `cpu/` and `gpu/`.

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
serial_inputs=("$RESULTS_DIR"/cpu/*/serial-*.csv)
"$PYTHON" scripts/plot_scaling.py --serial-only \
  --input "${serial_inputs[@]}" \
  --output-dir "$RESULTS_DIR/cpu/serial-figures"
```

Repeated-runtime figures use median times with min/max bars; summaries are CSVs. Speedup and efficiency plots show ratios of median times without min/max bars. The serial plotter also writes input metadata JSON. Time, size and resource axes are logarithmic; speedup/efficiency values are linear. The accuracy figure instead shows errors from converged refinement solves, not repeated-runtime statistics.

Strong speedup is `T(P0)/T(P)`; efficiency divides it by `P/P0`. Weak efficiency is `Tweak(P0)/Tweak(P)`. Each dimension/overlap series uses its smallest measured resource count: normally one CPU rank or one GPU/eight ranks. Sharing uses the one-rank/one-GPU baseline and has no hardware-efficiency curve.

Strong plots show runtime/speedup/efficiency, weak plots runtime/efficiency, and sharing plots runtime/speedup. Serial plots use unknowns/nonzeros. Weak runtime-versus-size plots are omitted, and obsolete ones are removed on rerun. A fixed-resource parallel size sweep is outside this protocol.

### Compare serial, MPI CPU and MPI+CUDA

Use the serial size-sweep, **strong-scaling** and GPU-sharing CSVs from the same study. The comparison selects matching global grids (4096² and 256³ by default), requires five distinct repeats, and checks problem, iteration count and warm-up consistency. It rejects weak-scaling inputs. It does not run any solver.

```bash
comparison_inputs=("$RESULTS_DIR"/cpu/*/serial-*.csv
                   "$RESULTS_DIR"/cpu/*/strong-*.csv
                   "$RESULTS_DIR"/gpu/*/strong-*.csv
                   "$RESULTS_DIR"/gpu/*/sharing-*.csv)
"$PYTHON" scripts/plot_comparison.py --input "${comparison_inputs[@]}" \
  --iterations "$ITERATIONS" --repeats "$REPEATS" --size-2d 4096 --size-3d 256 \
  --output-dir "$RESULTS_DIR/comparison"
```

Use `shopt -s nullglob` as above. Outputs are four `comparison_{runtime,speedup}_overlap{0,1}.png` figures and `comparison_summary.csv`. Figures have 2D/3D panels with shared y scales; execution configurations are categorical. Only measured configurations appear.

If strong and sharing inputs contain the same GPU/rank configuration, strong measurements take precedence. Repeats are checked independently and never pooled; settings must match. `source_experiment` records the selected input. Acceleration is matched serial median time divided by parallel median time. CPU serial-relative efficiency divides acceleration by rank count. GPU acceleration compares different hardware; GPU scaling efficiency still uses the separate one-GPU baseline.

## Supplementary layout and transport studies

These are separate **2,000-iteration, five-repeat** studies, with three warm-up iterations. Do not pool them with the 1,000-iteration resource-scaling study. Use a new output directory and an allocation label identifying the actual hardware/session. The consolidated entry point requires no project scheduler files. Run each block only inside the allocation specified for it.

```bash
LAYOUT_RESULTS="results/layouts"
mkdir -p "$LAYOUT_RESULTS"
# PYTHON is the selected interpreter from the common-settings section.
```

### CPU: 16 bound ranks on one host

Obtain 16 distinct allocated OS CPUs on one Linux host. The binding helper checks membership in the inherited scheduler CPU set, then binds each local MPI rank to one selected CPU. This is distinct from the unbound main scaling study.

```bash
export POISSON_CPUSET="$("$PYTHON" scripts/layout_study.py bind --print-cpuset)"
CPU_LABEL="cpu16-$(hostname)-SESSION_NAME"
CPU_LAYOUT=("$PYTHON" scripts/layout_study.py benchmark
  --exe build-cpu/executables/poisson_mpi --mode mpi-cpu
  --iterations 2000 --repeats 5 --communication host --bind-cpus
  --allocation-label "$CPU_LABEL"
  --launcher-arg=--map-by --launcher-arg=ppr:16:node
  --launcher-arg=--bind-to --launcher-arg=none)
"${CPU_LAYOUT[@]}" --dim 2 --process-grids 2x8 4x4 8x2 \
  --problem-sizes 128x128 512x512 1024x1024 \
  --output "$LAYOUT_RESULTS/cpu2d-layout.csv"
"${CPU_LAYOUT[@]}" --dim 3 --process-grids 2x8x1 4x4x1 8x2x1 \
  --problem-sizes 32x32x32 64x64x64 128x128x128 \
  --output "$LAYOUT_RESULTS/cpu3d-layout.csv"
```

Replace `SESSION_NAME` for each independent allocation. These matrices produce 90 records per dimension, 180 CPU records in total. The collector reverses the entire matched configuration order on alternate repeats. If the site's optional HCOLL/UCC collectives require disabling, append the same Open MPI MCA options used in the main CPU section. This does not change the point-to-point transport. Use `--dry-run` to inspect a plan without executing solvers or creating timings.

### GPU: one A40, eight ranks, plus a separate four-rank control

Use one allocated GPU accessible to all ranks on one host. For the eight-rank study, compare layouts, both host-staged/CUDA-aware transfers, and overlap off/on:

```bash
GPU_LABEL="gpu8-$(hostname)-SESSION_NAME"
GPU_LAYOUT=("$PYTHON" scripts/layout_study.py benchmark
  --exe build-gpu/executables/poisson_mpi --mode mpi-cuda
  --iterations 2000 --repeats 5 --expected-gpus 1
  --communication host cuda-aware --allocation-label "$GPU_LABEL"
  --launcher-arg=--map-by --launcher-arg=ppr:8:node
  --launcher-arg=--bind-to --launcher-arg=none)
"${GPU_LAYOUT[@]}" --dim 2 --process-grids 2x4 4x2 \
  --problem-sizes 256x256 1024x1024 \
  --output "$LAYOUT_RESULTS/gpu8-2d-layout.csv"
"${GPU_LAYOUT[@]}" --dim 3 --process-grids 2x4x1 4x2x1 \
  --problem-sizes 32x32x32 \
  --output "$LAYOUT_RESULTS/gpu8-3d-layout.csv"
```

These complete matrices contain 80 and 40 records, respectively. The separate four-rank 2x2 control at 256²/1024² contributes another 40 records:

```bash
"$PYTHON" scripts/layout_study.py benchmark \
  --exe build-gpu/executables/poisson_mpi --mode mpi-cuda --dim 2 \
  --process-grids 2x2 --problem-sizes 256x256 1024x1024 \
  --communication host cuda-aware --iterations 2000 --repeats 5 --expected-gpus 1 \
  --allocation-label "gpu4-$(hostname)-SEPARATE_SESSION_NAME" \
  --launcher-arg=--map-by --launcher-arg=ppr:4:node \
  --launcher-arg=--bind-to --launcher-arg=none \
  --output "$LAYOUT_RESULTS/gpu4-control.csv"
```

The report's accepted total is 160 GPU records. The incomplete 128³ sweep is not included in these commands. `--expected-gpus 1` and stable device/mapping checks prevent accidental pooling of changing GPU allocations.

### Audit and plot accepted complete matrices

For example, audit the eight-rank 2D GPU matrix before plotting:

```bash
"$PYTHON" scripts/layout_study.py audit \
  --input "$LAYOUT_RESULTS/gpu8-2d-layout.csv" --mode mpi-cuda --dim 2 \
  --process-grids 2x4 4x2 --problem-sizes 256x256 1024x1024 \
  --communication host cuda-aware --iterations 2000 --repeats 5 \
  --output "$LAYOUT_RESULTS/gpu8-2d-audit.json"
"$PYTHON" scripts/layout_study.py plot \
  --input "$LAYOUT_RESULTS/gpu8-2d-layout.csv" "$LAYOUT_RESULTS/gpu8-3d-layout.csv" \
  --output-dir "$LAYOUT_RESULTS/gpu8-figures"
"$PYTHON" scripts/layout_study.py transport \
  --input "$LAYOUT_RESULTS/gpu8-2d-layout.csv" "$LAYOUT_RESULTS/gpu8-3d-layout.csv" \
  --output "$LAYOUT_RESULTS/gpu8-transport-summary.csv"
"$PYTHON" scripts/layout_study.py plot \
  --input "$LAYOUT_RESULTS/cpu2d-layout.csv" "$LAYOUT_RESULTS/cpu3d-layout.csv" \
  --overview-sizes 1024 128 \
  --output-dir "$LAYOUT_RESULTS/cpu16-figures"
```

Audit the other CSVs with their corresponding exact sizes, grids and transports. The audit requires exactly five repeats per requested case; partial CSVs cannot stand in for a complete matrix. It verifies numerical consistency and, with `--job-log PATH`, can inspect privately retained binding/completion records from the historical CPU experiment. Logs are not final-submission artifacts.

Layout plots preserve each allocation, device, problem size, process grid and transport. They summarize min/median/max runtimes and the **median of paired T(no overlap)/T(overlap) ratios**. Transport analysis uses the median paired host/CUDA-aware ratio. These are not generally ratios of independent medians. Outputs include `layout_summary.csv`, PNGs and `figure_index.csv`.

`--overview-sizes 1024 128` additionally creates `layout_dimensions.png` for Figure 14: 2D at 1024² and 3D at 128³, side by side. GPU plots create `communication_*.png` with host staging and CUDA-aware MPI side by side; `communication_figure_index.csv` identifies each size and configuration. Figures 15 and 16 use the eight-rank 1024² and 32³ cases, respectively. Point these commands at the corresponding job-ID CSVs when using the retained study.

### Analysis safeguards

The main collector supports `--problem`, `--warmup`, `--label` and `--timeout`, verifies returned settings, and writes executable hashes and invocation provenance alongside its CSV. Keep those provenance files privately. The resource-scaling plotter keeps labels, warm-up, device and block-size series separate; it rejects mixed process-grid shapes. Optional `--serial-reference FILE...` computes acceleration only from serial measurements matching grid/problem/iterations/warm-up/label. Its existing MPI-relative speedup/efficiency definitions remain unchanged.

## Results organization and transfer

Keep independent studies separate. A useful local layout is:

```text
results/
├── performance/          main 1,000-iteration/five-repeat measurements
│ ├── cpu/                per-core-count CSVs, serial CSVs and figures/
│ ├── gpu/                per-GPU-count/sharing CSVs and figures/
│ └── comparison/         matched serial/CPU/GPU figures and summary
├── layouts/              supplementary 2,000-iteration studies and accepted analyses
├── accuracy/             converged refinement study
└── validation/           privately retained CPU/GPU check outputs
```

The commands above collect supplementary data under `results/layouts` by default; `LAYOUT_RESULTS` can select another directory. Existing downloaded datasets may use job-ID subdirectories; the main plotting globs accept these. Set `RESULTS_DIR` to the actual study root. Renaming a local folder does not rename remote data, and changing a new output default does not move existing results.

Reanalysis reads existing CSVs; it does not rerun the timing experiments. Keep original measurements and provenance, and do not merge the incomplete 29-record 128³ supplementary GPU sweep into the accepted 160-record GPU analysis. Validation primarily emits pass/fail output rather than performance CSVs. Store its logs privately, separately from source submission files.

### Transfer source and results

From your laptop Terminal, outside SSH, edit the account and paths and upload the source. Build Linux binaries on Vanda. These commands update files without deleting existing ones.

```bash
VANDA_USER="YOUR_USERID"
REMOTE="$VANDA_USER@vanda.nus.edu.sg"
REMOTE_DIR="/home/svu/$VANDA_USER/poisson-solver"
LOCAL_DIR="$HOME/Desktop/poisson-solver"
ssh "$REMOTE" "mkdir -p '$REMOTE_DIR'"
rsync -av --exclude='/build*/' --exclude='/.venv*/' --exclude='/results/' \
  --exclude='/outputs/' --exclude='/work/' --exclude='/tmp/' --exclude='/jobs/' \
  --exclude='/.git/' --exclude='__pycache__/' --exclude='.DS_Store' \
  --exclude='*.log' --exclude='*.o[0-9]*' --exclude='*.e[0-9]*' \
  "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"
ssh "$REMOTE"
```

In the remote shell, enter the uploaded project directory, obtain an allocation and follow the README build and experiment commands. Back on the laptop, download the completed study (change `performance` if you chose another study name):

```bash
mkdir -p "$LOCAL_DIR/results"
rsync -av --exclude='*.log' --exclude='*.o[0-9]*' --exclude='*.e[0-9]*' \
  "$REMOTE:$REMOTE_DIR/results/performance/" "$LOCAL_DIR/results/performance/"
rsync -av --exclude='*.log' --exclude='*.o[0-9]*' --exclude='*.e[0-9]*' \
  "$REMOTE:$REMOTE_DIR/results/accuracy/" "$LOCAL_DIR/results/accuracy/"
```

## Source organization

| Path | Purpose |
| --- | --- |
| `CMakeLists.txt` and subdirectory CMake files | CPU/CUDA/MPI builds and eight CTest groups |
| `src/` | Numerical problem, decomposition, CPU/CUDA backends, MPI exchange and timing driver |
| `executables/` | Serial and MPI entry points |
| `tests/test_numerics.cpp` | In-process numerical and Cartesian checks |
| `tests/validate_mpi.py` | Basic/extended independent executable validation |
| `tests/test_analysis_scripts.py` | 18 collection, plotting, layout and parser regression checks |
| `scripts/benchmark.py` | Main measured performance CSVs and invocation provenance |
| `scripts/layout_study.py` | Supplementary collection, binding, matrix audits, plots and paired ratios |
| `scripts/plot_scaling.py` | Scaling plots, optional matched serial reference and serial-only plotting |
| `scripts/plot_comparison.py` | Matched serial/CPU/GPU comparisons, including GPU sharing |
| `scripts/accuracy_study.py` | Converged sine refinement study and independent error checks |
| `scripts/solver_output.py` | Parse one solver summary while retaining launcher diagnostics |
| `namelist.txt` | Optional runtime settings loaded explicitly with `--config` |
| `requirements-analysis.txt` | Matplotlib dependency for plots and analysis tests |

