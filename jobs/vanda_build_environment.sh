#!/bin/bash
# Sourced only by build jobs, after the compiler/MPI environment and log setup.
echo 'Preparing build tools with jobs/vanda_build_environment.sh'
if [[ -z "${POISSON_CMAKE_MODULE:-}" && -x "$PWD/.venv-build/bin/cmake" ]]; then
    export PATH="$PWD/.venv-build/bin:$PATH"
fi
if [[ -n "${POISSON_CMAKE_MODULE:-}" ]] || ! command -v cmake >/dev/null 2>&1; then
    if ! module load "${POISSON_CMAKE_MODULE:-CMake}"; then
        echo 'Unable to load CMake. Check the available module names below.' >&2
        module avail CMake || true
        echo 'Resubmit with -v POISSON_CMAKE_MODULE=EXACT_MODULE_NAME.' >&2
        echo 'Alternatively install CMake on Vanda in this project directory:' >&2
        echo 'python3 -m venv .venv-build' >&2
        echo '.venv-build/bin/python -m pip install "cmake>=3.24,<4"' >&2
        exit 1
    fi
fi
for build_tool in cmake ctest make mpicxx mpiexec; do
    if ! command -v "$build_tool" >/dev/null 2>&1; then
        echo "Required build tool is unavailable: $build_tool" >&2
        echo 'Check the compiler/MPI and CMake modules before resubmitting.' >&2
        exit 1
    fi
done
cmake --version
command -v cmake

