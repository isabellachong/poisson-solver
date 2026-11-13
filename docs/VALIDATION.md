# Validation: 2D/3D Jacobi-Poisson solver

See [the main README](../README.md) for editable account/directory variables and
the Vanda job workflow. Run all commands below from the project root, not from
`docs/`.

## Study settings and results locations

Use the same settings as the README. In a Vanda shell:

```bash
export VANDA_USER="$(id -un)"
export VANDA_PROJECT_DIR="$HOME/dsa5208-project3"
export RESULTS_DIR="results/performance"
cd "$VANDA_PROJECT_DIR"
```

Change `VANDA_PROJECT_DIR` and `RESULTS_DIR` for your installation. Laptop transfer
commands in the README also use editable `VANDA_HOST` and `LOCAL_PROJECT_DIR`.
No personal account name or absolute laptop directory is required here.

The complete performance submission uses **1,000 iterations and five repeats**:

```bash
POISSON_RESULTS_DIR="$RESULTS_DIR" bash jobs/submit_1000x5.sh
```

Submit this study once, after the build/validation jobs pass. The helper refuses
an existing study directory. Its default output layout is:

```text
results/performance/
  job_ids.tsv
  results_roots.tsv
  submission_commands.sh
  cpu/cpu-scale-JOB_ID/    CPU MPI strong/weak CSVs and job.log
  cpu/serial-JOB_ID/       serial size-sweep CSVs and job.log
  cpu/figures/            CPU/serial PNGs and summary.csv
  gpu/scale-JOB_ID/        GPU scaling or sharing CSVs and job.log
  gpu/figures/            GPU PNGs and summary.csv
```

`JOB_ID` is the full PBS ID, including its server suffix. The separate accuracy
job writes to `results/accuracy/accuracy-JOB_ID/study/`. It uses converged solves,
not the fixed iteration count of the performance study.

Existing queued jobs retain their submitted paths. Rename an existing study
only after all its measurement and plotting jobs finish, following the guarded
rename instructions in the README. Preserve original paths in saved logs and
manifests as provenance.

## Evidence and limitations

This records local checks on macOS ARM64 and user-reported cluster checks.
Report-style serial figures were generated locally from the user's downloaded
Vanda measurements; these are not local solver timings. Source support for an
execution mode is distinct from having verified that mode on hardware.

| Check | Status |
| --- | --- |
| CMake CPU build: serial, MPI runtime, and MPI CPU executable | Passed (AppleClang 21, CMake 4.4.5, Open MPI 5.0.11) |
| Seven focused CTest groups | Passed |
| Serial 2D/3D executable versus independent Python reference | Passed |
| Initial convergence and iteration-limit exit behaviour | Passed |
| Serial benchmark collector smoke run | Passed; tiny local checks, not report performance measurements |
| Python helper-script syntax | Checked |
| Vanda PBS template syntax, rank/workload tables, and command construction | Checked locally; orchestration used command stubs and generated no timing data |
| CPU PBS workflows at 1/4/8/16/32/64/128/256 ranks and serial sweep selection | Checked with command stubs, including validation grid sizes, per-core weak work, binding arguments, and plot inputs |
| Configurable five-repeat submission results path | Local stub checks passed all 14 submissions with default results/performance and a custom study directory, plot dependencies, existing-directory protection and partial-rejection handling; these checks did not submit real jobs |
| Configurable independent validation grid and one-rank dispatch | Serial checks passed at default and 33x35x17 sizes; argument/dispatch checks passed without MPI execution |
| CMake environment preparation for PBS build jobs | Checked existing tools, default/versioned module loading, unavailable modules, and missing build tools with local command stubs |
| Vanda CPU build and CTest, user-reported job 1541774 | Build and all seven tests passed; actual MPI executable validation failed before launch in Open MPI core binding |
| Vanda CPU job 1541810, user-reported output | Build/CTest passed and the four-rank MPI solver returned a 2D result; its residual/max/RMS errors match the local serial case. Validation stopped because UCX stdout warnings were parsed as CSV |
| Vanda CPU independent executable validation, user confirmed | All eight executable checks passed after the parser fix, using the NVIDIA HPC-X/Open MPI stack and build-cpu |
| Solver summary parsing with launcher diagnostics | Five regression groups, serial executable validation, benchmark collection, and CPU/GPU mapping preflight passed locally with injected diagnostics |
| Actual multi-process MPI execution | Blocked locally by socket permissions; user-supplied Vanda logs show CPU validation and experiments through 256 ranks |
| CUDA configuration/compilation | NVCC is unavailable locally; the user built and ran the CUDA executable on Vanda |
| GPU numerical results and stream/event ordering | Vanda executable checks passed in the user-supplied logs; sanitizer and profiler verification of asynchronous behaviour remain separate tasks |
| One-GPU sharing and multi-GPU mapping/communication | Downloaded Vanda measurements include fixed-one-GPU sharing and two GPUs with 16 ranks, maximum eight ranks per GPU |
| Combined serial report figure | Generated and visually checked locally with Matplotlib 3.10.9 from Vanda job 1541919; 36 raw timings, 12 grid summaries, 1,000 iterations and three repeats verified |
| Dedicated spatial-accuracy study | Six local serial sine solves at N=7/15/31 in 2D/3D converged, passed independent finite-step eigenmode owned-value and max/RMS-error comparisons, and gave observed orders about 2.008/2.002. The logarithmic PNG was visually checked. The user subsequently reported Vanda accuracy job 1546292 finished and downloaded its study; its cluster output has not been independently inspected here |
| Report-style CPU/GPU scaling figures | Rendered and visually checked locally from completed downloaded Vanda CSVs through 256 CPU ranks and two GPUs, plus one-GPU sharing. All 104 median/min/max and speedup/efficiency rows match the raw timings. Axis labels/scales and weak-size annotations were checked on 34 panels |
| Multi-core/multi-GPU scaling results | User supplied the completed serial, CPU 1-256-rank, one-/two-GPU and sharing results; the local plotting checks use these measurements rather than generated timing data |

The recorded job IDs identify historical evidence, not jobs a reader should
reuse. The three-repeat plotting checks above concern the earlier completed
study. They do not establish completion of the subsequent five-repeat study;
check that study's own manifest, exit statuses and CSVs before reporting results.

[jobs/README.md](../jobs/README.md) and [jobs/README-CPU.md](../jobs/README-CPU.md) describe the prepared MPI+CUDA, CPU MPI,
and serial PBS jobs. Their queue limits, modules, resource placement, shared-GPU
access, and actual distributed execution require checks on Vanda.

MPI job templates now default to `POISSON_MPI_BIND_TO=none` after the reported
`hwloc_set_cpubind` failure. This uses `--bind-to none` for Open MPI and disables
pinning for Intel MPI. The user subsequently completed Vanda CPU runs with this policy.
The validator now creates its temporary solution directory under the working
directory, so the PBS jobs' shared project path is used across nodes.
The shared Python summary parser now accepts UCX diagnostics around the CSV
header/result, preserves diagnostics on stderr, and rejects ambiguous summaries.
The user subsequently confirmed all eight CPU executable checks passed. CPU job
defaults now match that validated stack and build directory. Subsequent
downloaded results include the completed multi-node CPU and two-GPU runs.

## Plot validation

The dedicated `scripts/plot_serial.py` reproduces the two-panel serial figure
style in Member 1's report using the current run's own measurements. It plots
runtime against unknown count and implicit matrix nonzeros, overlays 2D/3D,
uses median/min-max error bars, and labels the actual iteration/repetition counts.
Input hashes accompany the PNG and summary CSV. SVG and caption generation
were removed at the user's request.
Local checks rejected mixed iteration counts, nonfinite timings, incorrect
stencil counts and duplicate repeats. The final PNG was inspected for legibility
and axis limits. Its dotted references are anchored linear-work illustrations,
not fitted predictions or inferred timings.

CPU scaling resource axes now read `CPU count` for this one-core-per-rank study.
Time, problem-size and CPU/resource-count axes are logarithmic; speedup and
efficiency values remain linear. GPU-sharing process counts retain their MPI
rank label, since they do not represent allocated CPU-core counts. Additional
weak-scaling figures plot runtime versus global unknowns and matrix nonzeros,
and explicitly state that hardware also increases. A separate parallel
runtime-versus-problem-size sweep at fixed CPU/GPU count was not measured.

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
python3 tests/test_solver_output.py
```

## Reproduce cluster checks

The prepared jobs handle the module stack, launcher arguments and shared project
directory. Submit them from the project root with plain `qsub`:

```bash
qsub jobs/vanda_cpu_build_validate.txt
qsub jobs/vanda_build_validate.txt
qsub -v POISSON_RESULTS_ROOT="$RESULTS_DIR/cpu" jobs/vanda_cpu_multinode_check.txt
qsub jobs/vanda_accuracy.txt
```

The multi-node and accuracy jobs require the CPU build to have completed
successfully; accuracy also requires the plotting environment described in the
README. Check `qstat -fx JOB_ID` for `Exit_status = 0` and inspect each job's log.
The CPU build log is `results/vanda-cpu/cpu-check-JOB_ID/job.log`; the GPU build
log is `results/vanda/check-JOB_ID/job.log`. The two-node diagnostic log is
`$RESULTS_DIR/cpu/cpu-multinode-check-JOB_ID/job.log`, and the accuracy log is
`results/accuracy/accuracy-JOB_ID/job.log`.

The commands below are manual alternatives inside suitable allocations. On
Vanda, prefer the job scripts so the tested launcher settings are applied.

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
to establish memory correctness and actual overlap. For performance, collect
the matched 1,000-iteration/five-repeat serial, CPU MPI, GPU-sharing and multi-GPU
study described above. Report the actual CPU,
GPU model/count, mapping, compiler/toolkit/MPI versions, and MPS status. Keep all
comparisons matched in problem definition, iteration count, and timing scope.

Install plotting dependencies in the group's analysis environment with
`python3 -m pip install -r requirements-analysis.txt`, then render figures from
real benchmark CSVs. These plots are not substitutes for running the experiments.
