#include "backend.hpp"
#include "config.hpp"
#include "io_utils.hpp"
#include "mpi_runtime.hpp"
#include "problem.hpp"
#include "solver.hpp"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>
int main(int argc,char* argv[]) {
    bool initialized=false;
    int rank=0;
    try {
        if(poisson::help_requested(argc,argv)) {
            const char* launcher_rank=std::getenv("OMPI_COMM_WORLD_RANK");
            if(!launcher_rank || std::string(launcher_rank)=="0") poisson::print_usage(std::cout);
            return 0;
        }
        const auto c=poisson::parse_arguments(argc,argv);
        poisson::Device device;
#ifdef POISSON_USE_CUDA
        // Select before MPI_Init: some CUDA-aware MPI transports create CUDA state there.
        device=poisson::prepare_cuda_device();
        constexpr bool gpu=true;
#else
        constexpr bool gpu=false;
#endif
        int provided=0;
        if(MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided)!=MPI_SUCCESS) return 1;
        initialized=true;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD,MPI_ERRORS_RETURN);
        if(provided<MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED is unavailable");
        int status=0;
        {
            const double start=MPI_Wtime();
            poisson::MpiRuntime runtime(c,device,gpu);
            auto fields=poisson::initialize_fields(runtime.grid(),c.problem);
            std::unique_ptr<poisson::Backend> backend;
#ifdef POISSON_USE_CUDA
            backend=poisson::make_cuda_backend(runtime.grid(),runtime.active_faces(),fields,c);
#else
            backend=poisson::make_cpu_backend(runtime.grid(),runtime.active_faces(),fields);
#endif
            const auto result=poisson::solve(c,std::move(fields),*backend,runtime,runtime.info(),start);
            if(rank==0) poisson::write_summary(std::cout,c,runtime.info(),result);
            status=c.check_every && !result.converged ? 2 : 0;
        }
        MPI_Finalize(); return status;
    } catch(const std::exception& e) {
        std::cerr << "rank " << rank << ": " << e.what() << '\n';
        if(initialized) MPI_Abort(MPI_COMM_WORLD,1);
        return 1;
    }
}
