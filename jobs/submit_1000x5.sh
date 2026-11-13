#!/bin/bash
# Run with bash on the Vanda login node, not with qsub. This submits PBS jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v qsub >/dev/null 2>&1; then
    echo 'qsub is unavailable. Run this helper on the Vanda login node.' >&2
    exit 1
fi
for script in vanda_cpu_scaling.txt vanda_scaling.txt vanda_serial.txt vanda_plot.txt \
              vanda_environment.sh vanda_cpu_launcher.sh; do
    if [[ ! -r "jobs/$script" ]]; then
        echo "Missing job file: jobs/$script" >&2
        exit 1
    fi
done
for executable in build-cpu/executables/poisson_serial build-cpu/executables/poisson_mpi \
                  build-gpu/executables/poisson_serial build-gpu/executables/poisson_mpi \
                  .venv-analysis/bin/python; do
    if [[ ! -x "$executable" ]]; then
        echo "Missing validated executable or plotting Python: $executable" >&2
        exit 1
    fi
done

run_root="${POISSON_RESULTS_DIR:-results/performance}"
if [[ "$run_root" == *','* || "$run_root" == *$'\n'* ]]; then
    echo 'Choose a results directory without commas or newlines (PBS -v uses commas).' >&2
    exit 1
fi
if [[ -e "$run_root" ]]; then
    echo "Results directory already exists: $run_root" >&2
    echo 'Choose another directory with POISSON_RESULTS_DIR to keep studies separate.' >&2
    exit 1
fi
mkdir -p "$(dirname "$run_root")"
# Creating the study directory separately also prevents concurrent submissions
# from sharing a manifest and accidentally merging repeated measurements.
mkdir "$run_root"
cpu_results="$run_root/cpu"
gpu_results="$run_root/gpu"
mkdir -p "$cpu_results" "$gpu_results"
manifest="$run_root/job_ids.tsv"
commands="$run_root/submission_commands.sh"
printf 'category\tconfiguration\tjob_id\n' > "$manifest"
printf '#!/bin/bash\n# Submission record; inspect existing jobs before reusing these commands.\n' > "$commands"
printf 'cpu_results\t%s\ngpu_results\t%s\n' "$cpu_results" "$gpu_results" \
    > "$run_root/results_roots.tsv"

submission_count=0
trap 'poisson_submit_status=$?; if (( poisson_submit_status != 0 )); then
    echo "Submission stopped. Already accepted jobs remain queued; inspect $manifest before submitting the remainder." >&2
fi' EXIT

submit_job() {
    local category="$1" configuration="$2" job_id
    shift 2
    printf '%q ' qsub "$@" >> "$commands"
    printf '\n' >> "$commands"
    if ! job_id=$(qsub "$@"); then
        echo "qsub rejected $configuration." >&2
        return 1
    fi
    if [[ ! "$job_id" =~ ^[0-9]+(\.[A-Za-z0-9._-]+)?$ ]]; then
        echo "Unexpected qsub response for $configuration: $job_id" >&2
        echo 'Inspect qstat before retrying; PBS may have accepted this submission.' >&2
        return 1
    fi
    printf '%s\t%s\t%s\n' "$category" "$configuration" "$job_id" >> "$manifest"
    submission_count=$((submission_count+1))
    poisson_submitted_job_id="$job_id"
    printf '%-22s %s\n' "$configuration" "$job_id"
}

common_env='POISSON_MODULE=nvhpc-hpcx-cuda12/26.1,POISSON_MPI_BIND_TO=none,POISSON_ITERATIONS=1000,POISSON_REPEATS=5'
cpu_env="$common_env,POISSON_BUILD_DIR=build-cpu,POISSON_RESULTS_ROOT=$cpu_results,POISSON_CPU_STRONG_2D=4096,POISSON_CPU_STRONG_3D=256,POISSON_CPU_WEAK_2D=256,POISSON_CPU_WEAK_3D=32"
gpu_env="$common_env,POISSON_BUILD_DIR=build-gpu,POISSON_RESULTS_ROOT=$gpu_results,POISSON_STRONG_2D=4096,POISSON_STRONG_3D=256,POISSON_WEAK_2D=1024,POISSON_WEAK_3D=128,POISSON_COMMUNICATION=cuda-aware"

echo "Submitting 1,000-iteration, five-repeat experiments into $run_root"
cpu_jobs=()
for ranks in 1 4 8 16 32 64 128 256; do
    nodes=1
    ranks_per_node="$ranks"
    if (( ranks > 64 )); then
        nodes=$((ranks/64))
        ranks_per_node=64
    fi
    memory_gb="$ranks_per_node"
    if (( memory_gb < 4 )); then memory_gb=4; fi
    submit_job cpu "cpu_${ranks}_r5" \
        -N "cpu_p${ranks}_r5" \
        -l "select=$nodes:ncpus=$ranks_per_node:mpiprocs=$ranks_per_node:ompthreads=1:mem=${memory_gb}gb" \
        -l place=scatter -l walltime=01:00:00 \
        -v "$cpu_env,POISSON_CPU_RANKS=$ranks,POISSON_CPU_RANKS_PER_NODE=$ranks_per_node,POISSON_STUDY=all" \
        jobs/vanda_cpu_scaling.txt
    cpu_jobs+=("$poisson_submitted_job_id")
done

gpu_jobs=()
submit_job gpu gpu_1_r5 -N gpu_1_r5 \
    -l select=1:ncpus=36:mpiprocs=8:ompthreads=1:ngpus=1 \
    -l walltime=01:00:00 \
    -v "$gpu_env,POISSON_GPUS=1,POISSON_RANKS_PER_NODE=8,POISSON_STUDY=all" \
    jobs/vanda_scaling.txt
gpu_jobs+=("$poisson_submitted_job_id")

# Same validated two-GPU single-host layout as the completed experiment.
submit_job gpu gpu_2_r5 -N gpu_2_r5 \
    -l select=1:ncpus=72:mpiprocs=16:ompthreads=1:ngpus=2 \
    -l walltime=01:00:00 \
    -v "$gpu_env,POISSON_GPUS=2,POISSON_RANKS_PER_NODE=16,POISSON_STUDY=all" \
    jobs/vanda_scaling.txt
gpu_jobs+=("$poisson_submitted_job_id")

submit_job gpu gpu_sharing_r5 -N gpu_sharing_r5 \
    -l select=1:ncpus=36:mpiprocs=8:ompthreads=1:ngpus=1 \
    -l walltime=01:00:00 \
    -v "$gpu_env,POISSON_GPUS=1,POISSON_RANKS_PER_NODE=8,POISSON_STUDY=sharing" \
    jobs/vanda_scaling.txt
gpu_jobs+=("$poisson_submitted_job_id")

submit_job serial serial_r5 -N serial_r5 -l walltime=01:00:00 \
    -v "$cpu_env" jobs/vanda_serial.txt
serial_job="$poisson_submitted_job_id"

cpu_dependencies=$(IFS=:; printf '%s' "${cpu_jobs[*]}")
gpu_dependencies=$(IFS=:; printf '%s' "${gpu_jobs[*]}")
submit_job plot cpu_figures_r5 -N cpu_figures_r5 \
    -W "depend=afterok:$cpu_dependencies:$serial_job" \
    -v "POISSON_PYTHON=$PWD/.venv-analysis/bin/python,POISSON_RESULTS_ROOT=$cpu_results" \
    jobs/vanda_plot.txt
submit_job plot gpu_figures_r5 -N gpu_figures_r5 \
    -W "depend=afterok:$gpu_dependencies" \
    -v "POISSON_PYTHON=$PWD/.venv-analysis/bin/python,POISSON_RESULTS_ROOT=$gpu_results" \
    jobs/vanda_plot.txt

echo "Submitted $submission_count jobs. Job IDs: $manifest"
echo "CPU/serial results and figures: $cpu_results"
echo "GPU results and figures: $gpu_results"
