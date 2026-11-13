# Jacobi-Poisson in 2D and 3D

C++17 implementations of double-precision Jacobi iteration for
`-laplacian(u)=f` on the unit square/cube with prescribed Dirichlet boundaries.
The discretization uses a five-point stencil in 2D and seven-point stencil in 3D.

| Executable | Execution |
| --- | --- |
| `poisson_serial` | One CPU process, without MPI initialization |
| `poisson_mpi` in `build-cpu` | Single-threaded CPU computation per MPI rank |
| `poisson_mpi` in `build-gpu` | MPI decomposition with CUDA computation per rank |

All commands run from the project root. Run distributed/GPU commands inside an
appropriate allocation obtained through your cluster's normal resource system.
Allocate one CPU core per rank for CPU scaling, and the stated visible GPUs for
GPU experiments. No scheduler submission files are needed by these instructions.

## Requirements and build

Use a C++17 compiler, CMake >=3.20, Python 3, and an MPI C/C++ development
installation. GPU builds also require NVCC, a compatible NVIDIA driver and CUDA
Toolkit, plus CUDA-aware MPI when selecting that communication method.
Build and run with a consistent compiler/MPI stack.

The Vanda measurements used the `nvhpc-hpcx-cuda12/26.1` module. In an environment
where that module is provided, initialize it directly before building/running:

```bash
module purge
module load nvhpc-hpcx-cuda12/26.1
export OMP_NUM_THREADS=1
```

CPU build (serial and MPI CPU):

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DPOISSON_ENABLE_CUDA=OFF
cmake --build build-cpu --parallel 4
```

GPU build (serial and MPI+CUDA):

```bash
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
  -DPOISSON_ENABLE_CUDA=ON
cmake --build build-gpu --parallel 4
```

If required, set `-DCMAKE_CUDA_ARCHITECTURES=NN` to the allocated GPU's compute
capability. With CMake >=3.24 and a visible GPU, `native` can be used instead.
For a serial-only build, disable both `POISSON_ENABLE_MPI` and
`POISSON_ENABLE_CUDA`.

## Run the solvers

These small examples perform 1,000 fixed iterations:

```bash
# Serial 2D and 3D.
./build-cpu/executables/poisson_serial \
  --dim 2 --nx 128 --ny 128 --iterations 1000 --check-every 0
./build-cpu/executables/poisson_serial \
  --dim 3 --nx 32 --ny 32 --nz 32 --iterations 1000 --check-every 0

# CPU MPI: four ranks on four allocated cores.
mpiexec -n 4 ./build-cpu/executables/poisson_mpi \
  --dim 2 --nx 128 --ny 128 --px 2 --py 2 \
  --iterations 1000 --check-every 0 --communication host --overlap 1

# MPI+CUDA: eight ranks on one node with one allocated GPU visible to all ranks.
mpiexec -n 8 ./build-gpu/executables/poisson_mpi \
  --dim 3 --nx 64 --ny 64 --nz 64 --px 2 --py 2 --pz 2 \
  --iterations 1000 --check-every 0 --communication cuda-aware --overlap 1

# MPI+CUDA: sixteen ranks on one node with two allocated GPUs visible to all ranks.
mpiexec -n 16 ./build-gpu/executables/poisson_mpi \
  --dim 3 --nx 64 --ny 64 --nz 64 --px 4 --py 2 --pz 2 \
  --iterations 1000 --check-every 0 --communication cuda-aware --overlap 1
```

GPU selection is automatic: node-local rank modulo the number of visible CUDA
devices. The one-GPU example shares it among eight ranks; the two-GPU example
uses eight ranks per GPU when all ranks see the same two devices. Check
`gpus_used` and `max_ranks_per_gpu` in the result. Detection respects allocation
visibility; it does not request GPUs. `--communication host` provides GPU
computation with boundary transfers staged through host memory.

| Option | Meaning |
| --- | --- |
| `--dim 2` or `--dim 3` | Problem dimension |
| `--nx`, `--ny`, `--nz` | Global interior grid dimensions; 2D ignores `nz` |
| `--px`, `--py`, `--pz` | Process-grid dimensions, whose product must equal rank count |
| `--iterations K --check-every 0` | Exactly K measured iterations |
| `--check-every Q --rtol R --atol A` | Check true residual every Q iterations and stop at tolerance |
| `--overlap 0` or `1` | Disable/enable computation and boundary-exchange overlap |
| `--warmup K` | Untimed warm-up iterations followed by an initial-state reset |
| `--block-size N` | CUDA update/packing threads per block; default 256 |
| `--problem mixed` | Default manufactured problem; also `sine`, `polynomial`, `zero` |

Distributed process grids must split at least two spatial axes. One-rank
baselines are supported; two-rank decompositions are rejected. Stencil tiling is
not implemented. Run either executable with `--help` for all options.

## Validate and reproduce experiments

```bash
ctest --test-dir build-cpu --output-on-failure
python3 tests/validate_mpi.py --serial-only \
  --serial build-cpu/executables/poisson_serial
python3 tests/test_solver_output.py
```

See [docs/VALIDATION.md](docs/VALIDATION.md) for actual MPI/CUDA checks and the
converged spatial-accuracy study. The fixed-iteration performance runs do not
require the numerical solution to have converged.

[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) gives direct commands for:

- CPU strong/weak scaling at 1, 4, 8, 16, 32, 64, 128 and 256 cores.
- One-/two-GPU scaling with eight MPI ranks per GPU, and one-GPU sharing.
- Serial runtime versus problem size, all with 1,000 iterations and five repeats.
- Median runtime, speedup and efficiency plots from measured CSVs.
- Matched serial/MPI CPU/MPI+CUDA runtime and serial-relative acceleration figures.

Outputs default in these examples to the editable `RESULTS_DIR=results/performance`.
The benchmark creates new CSV files and refuses to overwrite an existing CSV;
choose a fresh results directory for another study. No results are fabricated.

## Project files

| Path | Purpose |
| --- | --- |
| `src/` | Problem definition, decomposition, CPU/CUDA backends, MPI exchange and solver |
| `executables/` | Serial and MPI entry points |
| `tests/` | CTest groups and independent executable validation |
| `scripts/` | Benchmark collection, CSV parsing, accuracy study and plotting |
| `scripts/plot_comparison.py` | Matched serial/CPU/GPU runtime and acceleration comparisons |
| `docs/IMPLEMENTATION.md` | Numerical algorithm, parallel design, timing and source reference |
| `docs/VALIDATION.md` | Correctness checks, available evidence and limitations |
| `docs/EXPERIMENTS.md` | Direct experiment and plotting commands |
| `docs/UPLOAD_TO_VANDA.md` | Editable upload/download commands |
| `namelist.txt` | Optional configuration loaded with `--config namelist.txt` |
| `requirements-analysis.txt` | Plotting dependencies |

## Submission and AI usage

Include the report PDF and source with this README. Keep builds, environments,
scheduler submission files, terminal transcripts and diagnostic logs outside the
final submission. Present runtime figures and validation findings in the report;
retain raw measurements privately for checking and analysis.

AI assistance contributed to implementation, tests and documentation. Describe
that assistance and the group's review and verification in the report. Local CPU
checks alone do not establish GPU correctness or distributed performance.
