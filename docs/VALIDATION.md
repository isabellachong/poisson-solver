# Validation of the 2D/3D solver

Run commands from the project root after the direct builds in
[the README](../README.md). Compiler/runtime preparation and resource allocation
are prerequisites; no project scheduler scripts are required.

## Checks and evidence

| Check | Available evidence and limits |
| --- | --- |
| CPU build and seven CTest groups | Passed locally on macOS ARM64 and in user-supplied Vanda results |
| Serial executable validation | Independent owned values, residuals, warm-up reset and stopping checks passed locally and on Vanda |
| CPU MPI validation | Eight executable checks passed on Vanda; a two-node check also passed after disabling optional HCOLL/UCC collectives |
| MPI CPU scaling | Earlier completed measurements covered 1–256 ranks; the subsequent five-repeat local dataset inspected so far covered 1–128 ranks. Completion of one study does not establish completion of another |
| CUDA build and numerical validation | Vanda logs supplied by the user show successful builds and independent GPU executable checks; NVCC is unavailable locally |
| GPU mapping and scaling | Earlier measurements include one-/two-GPU and fixed-one-GPU sharing. The user reported the five-repeat two-GPU and plotting runs completed; inspect those downloaded CSVs before making final numerical claims |
| Spatial accuracy | Downloaded Vanda refinement CSV contains six converged sine solves at N=7/15/31 in 2D/3D, all independent-reference checks passed; observed orders are approximately 2.008 and 2.002 |
| Plot calculations | Earlier downloaded three-repeat data were checked against all 104 summary rows; runtime, median/min/max, speedup/efficiency and axes were checked. This does not substitute for checking new inputs |
| Direct documentation commands | Bash syntax, local document links and 44 benchmark command constructions were checked with a recording stub: 32 CPU, eight GPU, two sharing and two serial commands. This check did not launch MPI/CUDA or generate performance measurements |
| Serial/CPU/GPU comparison figures | Four PNGs generated from downloaded five-repeat serial and strong-scaling CSVs; all 40 summary rows checked against raw medians, min/max and serial-relative acceleration. Duplicate repeats, mismatched settings, nonfinite runtimes and weak-scaling inputs were rejected; the runtime figure was visually checked |
| GPU memory safety and actual asynchronous overlap | Numerical tests do not establish these; sanitizer and timeline-profiler verification remain separate checks |

Actual local multi-process execution was restricted by socket permissions.
Distributed/CUDA evidence above comes from Vanda measurements and user-supplied
outputs, rather than invented or locally simulated GPU runtimes. Historical
three-repeat measurements must not be combined with the five-repeat study.

## Local CPU checks

```bash
ctest --test-dir build-cpu --output-on-failure
python3 tests/validate_mpi.py --serial-only \
  --serial build-cpu/executables/poisson_serial
python3 tests/test_solver_output.py
```

The seven CTest groups cover:

1. Decomposition: multidimensional partitions, uneven sizes and sparse-system counts.
2. Stencil: unequal spacing, negative-Laplacian sign and prescribed boundaries.
3. Regions: owned interior/boundary points updated exactly once, including thin partitions.
4. Exchange2D: packed faces reproduce full-grid CPU updates across local arrays.
5. Exchange3D: equivalent checks for six faces and 3D partitions.
6. Convergence: true residual stopping and second-order spatial accuracy.
7. Configuration: invalid settings, communication methods and dimensions.

The in-process exchange tests do not call MPI or execute CUDA kernels. Numerical
agreement alone cannot prove that communication and computation overlap in time.

## Actual MPI and CUDA checks

With at least eight allocated CPU cores, run the CPU executable validator:

```bash
python3 tests/validate_mpi.py \
  --serial build-cpu/executables/poisson_serial \
  --parallel build-cpu/executables/poisson_mpi --backend cpu
```

With one allocated GPU accessible to all eight ranks, run the GPU validator:

```bash
python3 tests/validate_mpi.py \
  --serial build-gpu/executables/poisson_serial \
  --parallel build-gpu/executables/poisson_mpi --backend cuda \
  --communication both
```

The GPU suite tests host-staged and CUDA-aware MPI, both overlap settings, and
owned values against an independent reference. Host-staged communication still
uses GPU computation. The default grids are 2x2 and 2x2x2. Repeat with additional
allocated GPUs and check the reported device counts and UUID mapping.

On the Vanda Open MPI stack, append these options to the CPU validator if needed
for the validated unbound/collective configuration:

```bash
  --launcher-arg=--bind-to --launcher-arg=none \
  --launcher-arg=--mca --launcher-arg=coll_hcoll_enable --launcher-arg=0 \
  --launcher-arg=--mca --launcher-arg=coll_ucc_enable --launcher-arg=0
```

For multiple hosts, also supply the allocation's host list through the launcher.
For example, where `PBS_NODEFILE` is defined, append
`--launcher-arg=--hostfile "--launcher-arg=$PBS_NODEFILE"`.
These are MPI options, not additional project files. On another MPI stack use its
appropriate arguments. Temporary per-rank solution files are created under the
shared working directory and checked for complete, unique ownership.

To validate a specific CPU process grid such as 256 ranks, use an allocation of
256 cores and pass `--points 33 35 17 --grid2d 16x16 --grid3d 8x8x4`, along with
the appropriate MPI placement arguments from [EXPERIMENTS.md](EXPERIMENTS.md).

## Converged spatial-accuracy study

Use one CPU core and a Python environment with the plotting dependencies:

```bash
python3 scripts/accuracy_study.py \
  --exe build-cpu/executables/poisson_serial \
  --output-dir results/accuracy/study --plot
```

The output directory must be new. The script solves the sine problem for N=7,
15 and 31 interior points per axis in both dimensions, with `rtol=1e-11`,
`atol=1e-12`, checks every ten iterations and at most 50,000 iterations.
An independent discrete sine-eigenmode reference checks all owned values and
reported maximum/RMS errors. The observed spatial order must lie between 1.9
and 2.1. The downloaded Vanda results used 320, 1,310 and 5,250 iterations,
respectively, in both dimensions; maximum errors were approximately 0.012951,
0.003219 and 0.000804.

Outputs are `refinement.csv`, `refinement.png`, `run.json` and owned solution CSVs.
These are analysis artifacts; present the accuracy figure and findings in the
report. The fixed-1,000-iteration performance experiments are separate and need
not have converged to this tolerance.

## Timing and figure validation

Follow [EXPERIMENTS.md](EXPERIMENTS.md) for matched timing inputs and direct
plotting commands. Check actual CPU/GPU counts, process grids, iteration count,
five distinct repeats per configuration and both overlap settings. Keep problem,
compiler/runtime environment, communication method and timing scope matched.
Strong/weak series use their own one-resource baselines; GPU sharing has no
hardware-efficiency curve. Weak runtime-versus-unknowns/nonzeros plots are excluded; hardware scaling
plots and the separate serial size plots are retained. A fixed-resource parallel
size sweep was not measured.

Retain raw evidence privately. The final submission should contain the requested
report and documented source, without diagnostic logs or terminal transcripts.
