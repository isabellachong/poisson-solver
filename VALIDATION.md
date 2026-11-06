# Validation status: fresh 2D/3D implementation

This records checks performed locally on macOS ARM64. No GPU or Vanda/Atlas
runtime figures were generated. Source support for an execution mode is distinct
from having verified that mode on hardware.

| Check | Status |
| --- | --- |
| CMake CPU build: serial, MPI runtime, and MPI CPU executable | Passed (AppleClang 21, CMake 4.4.5, Open MPI 5.0.11) |
| Seven focused CTest groups | Passed |
| Serial 2D/3D executable versus independent Python reference | Passed |
| Initial convergence and iteration-limit exit behaviour | Passed |
| Serial benchmark collector smoke run | Passed; tiny local checks, not report performance measurements |
| Python helper-script syntax | Checked |
| Actual multi-process MPI execution | Blocked locally: MPI socket creation returns operation-not-permitted |
| CUDA configuration/compilation | Not verified: NVCC/toolkit is not installed locally |
| GPU numerical results and stream/event ordering | Requires real GPU testing |
| One-GPU sharing and multi-GPU mapping/communication | Implemented, not hardware-verified here |
| Plot rendering | Not run here: matplotlib is not installed locally |
| Multi-core/multi-GPU scaling results | Not collected |

## Purpose of the seven CTest groups

1. **decomposition**: multidimensional shapes, rejection of 1D process layouts,
   uneven partitions, and implied sparse-system size/nonzero counts.
2. **stencil**: unequal spacings, the negative-Laplacian sign, prescribed nonzero
   physical boundaries, and correct initialization at internal rank boundaries.
3. **regions**: interior and boundary update regions cover each owned point once,
   including thin subdomains and 3D edges/corners.
4. **exchange2d**: packed faces transferred directly between local CPU arrays
   reproduce a full-grid calculation for uneven and thin 2x2 partitions.
5. **exchange3d**: the equivalent test for 2x2x2 partitions and all six faces.
6. **convergence**: true residual convergence and approximately second-order
   spatial accuracy in both dimensions.
7. **configuration**: input constraints, transfer-method settings, and dimension handling.

The in-process exchange checks do not execute MPI. Both overlap schedules are
checked numerically there, but CPU tests cannot verify asynchronous GPU execution.
The Python serial check uses an independently expressed finite-difference update,
reads actual per-rank output files, verifies owned values and final residual, and
checks that warm-up resets the starting approximation.

## Reproduce available checks

```sh
cmake -S . -B build-cpu -DPOISSON_ENABLE_CUDA=OFF
cmake --build build-cpu -j 4
ctest --test-dir build-cpu --output-on-failure
python3 tests/validate_mpi.py --serial-only
```

## Required cluster follow-up

Run in an allocated CPU job with enough slots:

```sh
python3 tests/validate_mpi.py --serial build-cpu/executables/poisson_serial --parallel build-cpu/executables/poisson_mpi --backend cpu
```

Build with CUDA in an appropriate allocation, then validate a single shared GPU:

```sh
cmake -S . -B build-gpu
cmake --build build-gpu -j 4
python3 tests/validate_mpi.py --serial build-gpu/executables/poisson_serial --parallel build-gpu/executables/poisson_mpi --backend cuda
```

The allocation must permit all ranks to access that GPU. The default suite uses
four ranks for 2D and eight for 3D, and checks host-staged/CUDA-aware communication
with/without overlap. Repeat with several allocated GPUs visible to the ranks;
automatic assignment uses all visible devices as needed. With enough GPUs for
one per rank, verify `max_ranks_per_gpu=1` in the summary. The emitted UUID mapping
and actual GPU/rank counts should match the allocation. No script enables MPS or changes cluster settings.

Run Compute Sanitizer on small actual GPU cases and profile a CUDA/MPI timeline
to establish memory correctness and actual overlap. Then collect repeated serial,
CPU MPI, GPU-sharing, and multi-GPU scaling measurements. Report the actual CPU,
GPU model/count, mapping, compiler/toolkit/MPI versions, and MPS status. Keep all
comparisons matched in problem definition, iteration count, and timing scope.

Install plotting dependencies in the group's analysis environment with
`python3 -m pip install -r requirements-analysis.txt`, then render figures from
real benchmark CSVs. These plots are not substitutes for running the experiments.
