# Implementation and manual execution reference

For the Vanda build, submission and results workflow, start with [the main README](../README.md#vanda-pbs-jobs).

## Build

Requirements: CMake >=3.20 and a C++17 compiler. MPI executables need an MPI C/C++
development installation. The CUDA executable additionally needs a compatible
NVIDIA driver, CUDA toolkit/NVCC, and allocated GPU access.

CPU build, including serial and MPI:

```sh
cmake -S . -B build-cpu -DPOISSON_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j 4
ctest --test-dir build-cpu --output-on-failure
```

GPU build, with CUDA enabled by default in `poisson_mpi`:

```sh
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpu -j 4
```

If the cluster requires an explicit GPU target, add
`-DCMAKE_CUDA_ARCHITECTURES=NN` with the compute capability of the allocated GPU
(for example, `80` means 8.0). Do not assume that example matches your GPU.
`-DMPI_CXX_COMPILER=/path/to/mpicxx` can help locate the intended MPI installation.
A serial-only build is possible with both `POISSON_ENABLE_MPI=OFF` and
`POISSON_ENABLE_CUDA=OFF`.

Lecture 12 lists Vanda's recommended CUDA-aware module as
`module load nvhpc-hpcx-cuda12/26.1`. Use the current cluster allocation and module
instructions when building/running there. The manual commands above do not
submit jobs; the Vanda workflow in the main README uses PBS job files and a submission helper.

## Basic runs

Run from the project directory, inside the appropriate CPU/GPU allocation:

```sh
# Serial 2D and 3D, fixed iteration counts.
./build-cpu/executables/poisson_serial --dim 2 --nx 128 --ny 128 --iterations 500
./build-cpu/executables/poisson_serial --dim 3 --nx 32 --ny 32 --nz 32 --iterations 500

# CPU MPI: four rectangular subdomains, then eight 3D blocks.
mpiexec -n 4 ./build-cpu/executables/poisson_mpi --dim 2 --nx 128 --ny 128 --px 2 --py 2
mpiexec -n 8 ./build-cpu/executables/poisson_mpi --dim 3 --nx 32 --ny 32 --nz 32 --px 2 --py 2 --pz 2

# Solve a 3D manufactured problem to a true residual tolerance.
mpiexec -n 8 ./build-cpu/executables/poisson_mpi --dim 3 --nx 31 --ny 31 --nz 31 --problem sine --iterations 20000 --check-every 20 --overlap 1
```

Dimensions count global **interior** points; physical boundary points are extra.
2D ignores `nz`. Process-grid dimensions can be omitted or set to zero to choose
them automatically. Explicit dimensions must multiply to the MPI rank count.

`--help` lists all options. `--config namelist.txt` explicitly loads key/value
settings; otherwise compiled defaults apply. Command-line values override file
values. Unknown settings, malformed numbers, and invalid process grids fail with
an error. `overlap` accepts `0` or `1`.

## One GPU and multiple GPUs

GPU selection occurs before `MPI_Init`, because some CUDA-aware MPI transports
create device resources during initialization. Open MPI, Intel MPI, MVAPICH, and
Slurm node-local-rank environment variables are recognized. A custom launcher can
set `POISSON_LOCAL_RANK` separately for each process. This program never sets
`CUDA_VISIBLE_DEVICES`; it respects the devices made visible by your allocation.

GPU assignment is automatic. Each process calls `cudaGetDeviceCount()` and
selects `node_local_rank % visible_device_count`. Ranks share GPUs when there
are more local ranks than visible devices. No GPU-assignment options are needed.

For eight ranks on a node where every rank sees the same device list:

| Visible GPUs | Automatic assignment |
| --- | --- |
| 1 | Eight ranks share one GPU |
| 2 | Four ranks per GPU |
| 4 | Two ranks per GPU |
| 8 | One rank per GPU |

If the scheduler exposes one distinct assigned GPU per rank, that GPU is logical
device 0 in each process, which is selected automatically. Detection finds visible
devices, not idle devices. To share exactly one GPU, ranks must be on that GPU's
node and have only that allocated GPU visible. If separate nodes each use one GPU,
the run uses several GPUs in total. Actual UUID mapping is printed to stderr and
actual device counts are included in the summary.

### Several ranks sharing one GPU

On a single node where one allocated GPU is visible to all ranks:

```sh
# A genuine 2x2 decomposition, all four ranks sharing the one visible GPU.
mpiexec -n 4 ./build-gpu/executables/poisson_mpi --dim 2 --nx 256 --ny 256 --px 2 --py 2 --overlap 0

# A 2x2x2 decomposition, all eight ranks sharing that GPU, with overlap.
mpiexec -n 8 ./build-gpu/executables/poisson_mpi --dim 3 --nx 64 --ny 64 --nz 64 --px 2 --py 2 --pz 2 --overlap 1
```

Sharing requires a cluster compute configuration that permits the processes to
use the assigned GPU. NVIDIA MPS may improve concurrency when processes
underutilize a GPU, but it is **not started or required by this code**. Follow the
cluster's MPS policy, if available, and record whether MPS was enabled. Without it,
separate CUDA contexts can be time-sliced. Correct shared execution does not imply
that kernels from different processes run simultaneously or that it is faster.

### Multiple GPUs

With four allocated GPUs available on the node, or distinct assigned GPUs visible
to the four ranks:

```sh
mpiexec -n 4 ./build-gpu/executables/poisson_mpi --dim 2 --nx 1024 --ny 1024 --px 2 --py 2 --overlap 1
```

With eight ranks and two GPUs visible to those ranks, round-robin mapping assigns
four ranks per GPU when the visible lists agree:

```sh
mpiexec -n 8 ./build-gpu/executables/poisson_mpi --dim 3 --nx 128 --ny 128 --nz 128 --px 2 --py 2 --pz 2 --overlap 1
```

A fully three-direction decomposition requires eight ranks, not eight GPUs.
With four ranks, a 3D problem can use `2x2x1`, which splits two directions.
Two-rank runs are deliberately rejected because they would split only one axis.

## Communication and overlap

The default CUDA transfer method is `--communication cuda-aware`. MPI sends and
receives contiguous **GPU buffers**. On Open MPI builds that expose the CUDA
support query, lack of CUDA-aware support is detected and reported before
communication. On other MPI implementations, verify CUDA-aware support yourself.

For a portable alternative, use `--communication host`. Boundary values are
copied through reusable pinned host buffers. This still computes on the GPU.
The CPU MPI executable always uses host buffers regardless of this CUDA option.

Each rank exchanges rows/columns in 2D and planes in 3D. Only face neighbours are
needed by these stencils. Noncontiguous boundary values are packed into contiguous
send buffers and received data is unpacked into the neighbouring-value storage.
Physical boundary values remain prescribed and are not MPI neighbours.

Both GPU schedules use the same update regions and kernel launches:

1. Post receives, pack outgoing boundary values, and complete packing (plus D2H
   staging in host mode) before posting sends.
2. With `overlap=1`, launch the strictly local interior update now.
3. Complete MPI requests; prepare received values by unpacking (plus H2D staging
   in host mode). Record a CUDA event on the boundary stream.
4. Make the compute stream wait for that event. With `overlap=0`, launch the
   interior update here. Then update the owned boundary regions.
5. Finish the compute stream and swap old/new solution pointers.

Owned edge/corner points belong to exactly one update region. Outgoing data is
ready before MPI reads it. Receives complete before unpacking. Event ordering
ensures boundary updates read current received values, and completing compute
makes communication buffers safe to reuse. The interior kernel reads owned data,
so it is independent of incoming boundary values.

The CPU waits in MPI while the interior GPU kernel may execute. Packing is not
overlapped with that kernel in this implementation. Actual overlap depends on
MPI progress, hardware, subdomain size, and GPU sharing; verify it using a
CUDA/MPI timeline profiler. There is no per-iteration global `MPI_Barrier`.

## Numerical problem and validation

The matrix is implicit. For dimension `d` and spacing `h[a]=1/(N[a]+1)`:

```text
next[i] = (f[i] + sum_a (old[i-e_a] + old[i+e_a])/h[a]^2)
          / (2 * sum_a 1/h[a]^2)
```

Interior values initially equal zero. Boundary values are initialized in both
solution arrays and remain fixed. Initialization uses global coordinates even
when a rank's local subdomain starts at a nonzero offset.

Manufactured problems:

| Name | Exact solution |
| --- | --- |
| `sine` | Product of `sin(pi*x[a])` in the active dimensions |
| `mixed` | The sine solution plus `0.25*sin(2*pi*x)*sin(3*pi*y)` (also multiplied by `sin(pi*z)` in 3D) |
| `polynomial` | `x+2*y` (+`3*z` in 3D) plus the product of `x[a]*(1-x[a])` |
| `zero` | Zero everywhere |

The RHS is generated analytically as the negative Laplacian. The exact solution
is used for boundaries and final error checks, not to initialize the interior.
The polynomial is also exact for the discrete second derivatives, up to roundoff.

Convergence uses `||f-A_h*u||_inf <= atol + rtol*||f||_inf`. Each GPU performs a
two-stage maximum reduction; MPI combines the resulting **host scalar** using
`MPI_Allreduce(MPI_MAX)`. This does not depend on device-buffer nonblocking
collectives. Neighbouring values are refreshed after a swap before evaluating
the current residual.

- `--check-every 0`: perform exactly the requested iterations, then evaluate
  the residual outside the timed solve. Use for primary scaling comparisons.
- Positive `check-every`: check the initial iterate, periodically during the
  solve, and at the iteration limit. Checks inside the loop are timed.
- Warm-up runs are untimed, followed by a reset to the initial problem.

Exit code 0 means fixed iterations completed or a tolerance solve converged;
2 means tolerance was not reached by the iteration limit; 1 means an error.

Seven focused CTest checks cover partitioning, stencil/boundary correctness,
nonoverlapping update regions, in-process boundary exchange in 2D and 3D,
second-order convergence, and invalid configuration. They do not require MPI
sockets or GPUs and do not establish actual MPI or CUDA execution correctness.

An independent Python numerical reference checks the executable's owned values,
global indexing, residual, warm-up reset, and stopping behaviour:

```sh
# Available locally without MPI communication.
python3 tests/validate_mpi.py --serial-only

# Real CPU MPI execution in an allocation with at least eight CPU slots.
python3 tests/validate_mpi.py --serial build-cpu/executables/poisson_serial --parallel build-cpu/executables/poisson_mpi --backend cpu

# Actual GPU kernels, both transports and overlap modes. One shared GPU suffices
# if the allocation permits eight processes to use it.
python3 tests/validate_mpi.py --serial build-gpu/executables/poisson_serial --parallel build-gpu/executables/poisson_mpi --backend cuda
```

For distinct GPUs, allocate enough devices and expose them consistently through
the launcher. Verify that `max_ranks_per_gpu` equals 1 in the CSV summary. The
default distributed validation grids are `2x2` and `2x2x2`;
`--grid3d 2x2x1` allows a four-rank 3D validation run. For launcher-specific
arguments, repeat `--launcher-arg=VALUE`. Also run GPU memory checks and profiling
on the actual cluster. See [VALIDATION.md](VALIDATION.md) for what was verified here.

## Timing and scaling

Rank 0 prints one CSV header and one record to stdout. Mapping/diagnostics go to
stderr. The CSV contains dimensions, unknowns, implied matrix nonzeros, process
grid, actual GPUs used, maximum ranks per GPU, numerical settings, errors,
iteration count, and maximum-across-ranks timings.

| Time | Contents |
| --- | --- |
| `setup_seconds` | Communicator/GPU mapping collection (MPI), local arrays, initialization, warm-up/reset, initial residual if enabled |
| `solve_seconds` | Iterations, communication, packing/unpacking, GPU completion, any in-loop convergence checks |
| `diagnostics_seconds` | Final residual if needed, solution download, final errors, optional solution CSV writing |

Configuration parsing, pre-MPI GPU selection, MPI initialization/finalization,
the start barrier, cleanup, and summary printing are excluded. These are solver
phase times, not end-to-end job wall time. Each reported phase is reduced using
the maximum rank time. The final GPU update has completed before solve timing ends.

The benchmarking script launches real executables inside an existing allocation,
alternates overlap order across repeats, and records results. It never simulates
GPU measurements or submits jobs. Existing CSVs are not overwritten.

```sh
# Serial runtime versus problem size; repeat with --dim 3 and suitable sizes.
python3 scripts/benchmark.py --exe build-cpu/executables/poisson_serial --mode serial --kind serial --dim 2 --sizes 64 128 256 512 --output results/serial2d.csv

# Strong scaling to 256 CPU ranks, with one CPU core allocated per rank.
python3 scripts/benchmark.py --exe build-cpu/executables/poisson_mpi --mode mpi-cpu --kind strong --dim 2 --process-grids 1x1 2x2 4x4 8x8 16x16 --nx 2048 --ny 2048 --output results/cpu2d-strong.csv

# Weak scaling: nx/ny here specify LOCAL counts; global counts grow with process grid.
python3 scripts/benchmark.py --exe build-cpu/executables/poisson_mpi --mode mpi-cpu --kind weak --dim 2 --process-grids 1x1 2x2 4x4 8x8 16x16 --nx 128 --ny 128 --output results/cpu2d-weak.csv

# With one allocated GPU visible: increase ranks while GPU count stays fixed.
# This is a GPU-sharing study, not GPU hardware scaling.
python3 scripts/benchmark.py --exe build-gpu/executables/poisson_mpi --mode mpi-cuda --kind sharing --dim 2 --process-grids 1x1 2x2 4x4 --nx 1024 --ny 1024 --output results/gpu-sharing.csv
```

Request and bind the required CPU/GPU resources using the cluster's instructions.
256 ranks require 256 allocated CPU cores for a 256-core study; do not oversubscribe
a smaller CPU allocation. GPU sharing is local to each GPU node, not remote GPU access.

For GPU strong scaling, keep global unknown count fixed while increasing
**physical GPUs actually used**. Either keep ranks/GPU constant, or keep the total
MPI rank count and decomposition constant while reducing sharing as GPUs increase.
Report which strategy was used. For GPU weak scaling, keep
unknowns/GPU and ranks/GPU fixed while increasing GPUs. The benchmark script
checks the measured rank/GPU ratio for weak-scaling runs. A sharing study checks
that measured GPU count stays fixed. With only one GPU, report shared-rank and
overlap performance, and state that multi-GPU scaling was not measured.

For example, a constant four ranks/GPU can use 2D process grids `2x2`, `2x4`, and
`4x4` with one, two, and four GPUs, respectively. Use allocations/device visibility
that actually provide those mappings and combine the collected CSVs. One rank/GPU
can instead use a one-rank baseline and four/eight/etc. ranks as resources permit.
Another valid strong-scaling experiment keeps eight ranks and a `2x2x2` grid fixed
while increasing visible allocated GPUs from one to two, four, and eight. This
measures acceleration as GPU sharing decreases. Collect one strong-scaling CSV
per allocation and combine them for plotting; keep global grid size unchanged.
For 3D CPU studies, possible grids include `1x1x1`, `2x2x2`, `4x4x4`, and
`8x8x4` (256 ranks). Noncubic grids are supported.

Plotting requires Python `matplotlib` in addition to the standard library:

```sh
python3 scripts/plot_scaling.py --input results/cpu2d-strong.csv results/cpu2d-weak.csv --output-dir results/figures
```

The script writes runtime plots, speedup/efficiency plots where meaningful, and a
median/min/max summary table. Time, problem-size, and resource-count axes use
logarithmic scales. Speedup is relative to the smallest resource count **in each
series**, identified in the table. For a one-core/one-GPU baseline this is the usual
strong-scaling speedup; otherwise interpret it relative to that larger baseline.
GPU sharing gets a relative-runtime speedup but no GPU hardware-efficiency figure.
GPU strong plots require constant ranks/GPU or constant total ranks; GPU weak
plots require constant ranks/GPU. CPU speedup uses the
one-rank MPI baseline when supplied; report serial-to-MPI speedup separately if
comparing against a separately measured serial reference.

## Sparse system size and memory

For 2D, `M=Nx*Ny` and `nnz=5*M-2*(Nx+Ny)` after eliminating prescribed boundary
unknowns. For 3D, `M=Nx*Ny*Nz` and
`nnz=7*M-2*(Nx*Ny+Nx*Nz+Ny*Nz)`. These counts are reported even though the matrix
is not stored. `64x64` in 2D and `16x16x16` in 3D each have 4096 unknowns but
20224 and 27136 nonzeros, respectively. Dimensionality also changes connectivity
and communication; explain those effects rather than attributing all runtime
differences to the number of nonzeros.

GPU storage starts with three double-precision local arrays (old, next, forcing),
plus received-neighbour storage, packed boundary buffers, reduction storage, and
per-process CUDA context overhead. Shared ranks compete for one GPU's memory and
bandwidth; sharing does not multiply available GPU memory. MPI boundary counts
larger than `INT_MAX` are rejected. Choose problems that fit actual allocations.

## Source organization and lecture paradigms

```text
CMakeLists.txt, namelist.txt     build and optional runtime configuration
src/types.hpp                  shared grid/configuration/result definitions
src/config.*                   settings and validation
src/problem.*                  stencils, forcing, boundaries, update regions
src/decomposition.*            compact process-grid selection and partitions
src/backend.hpp                common compute/memory interface
src/cpu_backend.cpp            serial and MPI CPU computation
src/cuda_kernels.*             CUDA updates, boundary packing, residual reduction
src/cuda_backend.cu            GPU selection, arrays, streams/events, transfer modes
src/runtime.hpp                serial execution/collective interface
src/mpi_runtime.*              Cartesian neighbours, MPI exchange/reductions, GPU accounting
src/solver.*                   common numerical iteration and convergence driver
src/io_utils.*                 CSV summaries and optional per-rank solutions
executables/                   serial and MPI entry points
tests/                         seven focused checks and executable validation
scripts/                       real benchmarking and scaling figures/tables
scripts/solver_output.py        CSV summary parsing with launcher diagnostics
scripts/plot_serial.py          combined serial report figure
scripts/accuracy_study.py       converged spatial-accuracy study
```

The main patterns are domain decomposition, data parallelism, stencil calculation,
explicit message passing, and collective reduction. CUDA threads share device
arrays; block-local shared memory supports the residual reduction. Overlap
schedules independent work while communication progresses. The layout follows
the supplied reference repository's reusable `src/` library and executable-driver
organization, without its unrelated numerical dependencies.

Optional per-rank solution files are enabled with `--solution-prefix results/u`;
create the parent directory first. Each file lists only owned points, with global
indices and coordinates. No global solution gather occurs in the iteration loop.
