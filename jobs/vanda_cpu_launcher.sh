#!/bin/bash
# Sourced after the common environment and ranks_per_node are set.
# The selected MPI must match the CPU build.
mpi_version=$(mpiexec --version)
case "$mpi_version" in
    *'Intel(R) MPI'*|*'Intel MPI'*)
        if [[ "$poisson_bind_to" == core ]]; then
            export I_MPI_PIN=on
            export I_MPI_PIN_DOMAIN=core
        else
            export I_MPI_PIN=off
        fi
        export I_MPI_PIN_RESPECT_CPUSET=on
        export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=off
        poisson_mpi_args=(-ppn "$ranks_per_node") ;;
    *'Open MPI'*|*OpenRTE*)
        # Disable optional HCOLL/UCC collectives after their UCX connection
        # setup failed on the user's two-node CPU allocations. Keep the
        # point-to-point transport unchanged; this does not force TCP.
        poisson_mpi_args=(--map-by "ppr:${ranks_per_node}:node" --bind-to "$poisson_bind_to"
                          --mca coll_hcoll_enable 0 --mca coll_ucc_enable 0)
        echo 'MPI CPU collectives: HCOLL and UCC disabled; default point-to-point transport retained'
        echo "MPI CPU transport environment: UCX_TLS=${UCX_TLS:-automatic}, UCX_NET_DEVICES=${UCX_NET_DEVICES:-automatic}, OMPI_MCA_pml=${OMPI_MCA_pml:-automatic}"
        # Some MPI builds do not discover the PBS multi-node allocation.
        # Use only the hosts/slots supplied for this job by PBS, including for
        # validation and benchmark subprocesses that reuse these arguments.
        if [[ -n "${PBS_NODEFILE:-}" ]]; then
            if [[ ! -r "$PBS_NODEFILE" || ! -s "$PBS_NODEFILE" ]]; then
                echo "PBS node file is unreadable or empty: $PBS_NODEFILE" >&2
                exit 1
            fi
            poisson_mpi_args=(--hostfile "$PBS_NODEFILE" "${poisson_mpi_args[@]}")
            echo "MPI allocated-host file: $PBS_NODEFILE"
        fi ;;
    *)
        echo "Unrecognized MPI launcher: $mpi_version" >&2
        exit 1 ;;
esac
echo "MPI launcher binding policy: $poisson_bind_to"
if command -v taskset >/dev/null 2>&1; then taskset -pc "$$" || true; fi
poisson_launcher_args=()
for launcher_argument in "${poisson_mpi_args[@]}"; do
    poisson_launcher_args+=("--launcher-arg=$launcher_argument")
done

