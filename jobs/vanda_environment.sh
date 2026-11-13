#!/bin/bash
# Sourced by the PBS jobs after PBS has selected the project directory.
if ! type module >/dev/null 2>&1 && [[ -r /app1/ebenv ]]; then
    source /app1/ebenv
fi
if ! type module >/dev/null 2>&1; then
    echo 'Module environment unavailable. Check hpc cpu or hpc gpu for the site setup.' >&2
    exit 1
fi
module purge
module load "${POISSON_MODULE:-nvhpc-hpcx-cuda12/26.1}"
export OMP_NUM_THREADS=1
poisson_python="${POISSON_PYTHON:-python3}"
poisson_build="${POISSON_BUILD_DIR:-build-gpu}"
poisson_results="${POISSON_RESULTS_ROOT:-results/vanda}"
poisson_bind_to="${POISSON_MPI_BIND_TO:-none}"
case "$poisson_bind_to" in
    none|core) ;;
    *) echo 'POISSON_MPI_BIND_TO must be none or core.' >&2; exit 1 ;;
esac
echo "MPI binding policy: $poisson_bind_to"
module list

