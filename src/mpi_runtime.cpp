#include "mpi_runtime.hpp"
#include "decomposition.hpp"
#include "problem.hpp"
#if defined(OPEN_MPI)
#include <mpi-ext.h>
#endif
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace poisson {
void mpi_check(int code, const char* operation) {
    if(code==MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING]; int length=0;
    MPI_Error_string(code,message,&length);
    throw std::runtime_error(std::string(operation)+": "+std::string(message,length));
}
MpiRuntime::MpiRuntime(const Config& c, const Device& device, bool gpu) {
    mpi_check(MPI_Comm_rank(MPI_COMM_WORLD,&info_.rank),"MPI_Comm_rank");
    mpi_check(MPI_Comm_size(MPI_COMM_WORLD,&info_.ranks),"MPI_Comm_size");
    info_.mode=gpu ? "mpi-cuda" : "mpi-cpu";
    info_.device_name=device.name;
    info_.process_grid=choose_process_grid(c,info_.ranks);
    int periodic[3]{0,0,0};
    mpi_check(MPI_Cart_create(MPI_COMM_WORLD,c.dimension,info_.process_grid.data(),periodic,0,&communicator_),"MPI_Cart_create");
    mpi_check(MPI_Comm_set_errhandler(communicator_,MPI_ERRORS_RETURN),"MPI_Comm_set_errhandler");
    Int3 coordinates{0,0,0};
    mpi_check(MPI_Cart_coords(communicator_,info_.rank,c.dimension,coordinates.data()),"MPI_Cart_coords");
    grid_=make_grid(c,info_.process_grid,coordinates);
    neighbours_.fill(MPI_PROC_NULL);
    for(int axis=0;axis<c.dimension;++axis) {
        // Find the ranks of the neighbouring processes along this axis.
        mpi_check(MPI_Cart_shift(communicator_,axis,1,&neighbours_[2*axis],&neighbours_[2*axis+1]),"MPI_Cart_shift");
        const auto count=face_size(grid_,2*axis);
        if(count>static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("A boundary message exceeds MPI int count; use smaller subdomains");
        counts_[2*axis]=counts_[2*axis+1]=static_cast<int>(count);
    }
    requests_.fill(MPI_REQUEST_NULL);
    if(gpu) {
#if defined(OPEN_MPI) && defined(MPIX_CUDA_AWARE_SUPPORT)
        if(c.communication=="cuda-aware") {
#if MPIX_CUDA_AWARE_SUPPORT
            if(!MPIX_Query_cuda_support()) throw std::runtime_error("This MPI installation reports no CUDA-aware support; use --communication host");
#else
            throw std::runtime_error("This MPI build has no CUDA-aware support; use --communication host");
#endif
        }
#endif
        char host[MPI_MAX_PROCESSOR_NAME]{}; int host_length=0;
        mpi_check(MPI_Get_processor_name(host,&host_length),"MPI_Get_processor_name");
        const std::string key=std::string(host,host_length)+"|"+device.uuid;
        constexpr int record_size=MPI_MAX_PROCESSOR_NAME+256;
        std::array<char,record_size> record{};
        if(key.size()>=record.size()) throw std::runtime_error("GPU mapping record is too long");
        std::copy(key.begin(),key.end(),record.begin());
        std::vector<char> records(static_cast<std::size_t>(info_.ranks)*record_size);
        mpi_check(MPI_Allgather(record.data(),record_size,MPI_CHAR,records.data(),record_size,MPI_CHAR,communicator_),"MPI_Allgather(GPU mapping)");
        std::map<std::string,int> use_count;
        for(int r=0;r<info_.ranks;++r) ++use_count[std::string(records.data()+static_cast<std::size_t>(r)*record_size)];
        info_.gpus_used=static_cast<int>(use_count.size());
        for(const auto& item:use_count) info_.max_ranks_per_gpu=std::max(info_.max_ranks_per_gpu,item.second);
        if(info_.rank==0) {
            std::cerr << "GPU mapping: " << info_.gpus_used << " device(s), maximum " << info_.max_ranks_per_gpu << " rank(s) per GPU\n";
            for(int r=0;r<info_.ranks;++r)
                std::cerr << "  rank " << r << ": " << records.data()+static_cast<std::size_t>(r)*record_size << '\n';
        }
    }
    if(info_.rank==0) std::cerr << "Process grid: " << info_.process_grid[0] << 'x' << info_.process_grid[1]
                              << (c.dimension==3 ? "x"+std::to_string(info_.process_grid[2]) : "") << '\n';
}
MpiRuntime::~MpiRuntime() { if(communicator_!=MPI_COMM_NULL) MPI_Comm_free(&communicator_); }
FaceMask MpiRuntime::active_faces() const {
    FaceMask mask{};
    for(int f=0;f<6;++f) mask[f]=neighbours_[f]!=MPI_PROC_NULL;
    return mask;
}
void MpiRuntime::begin_exchange(Backend& backend) {
    if(active_) throw std::logic_error("A boundary exchange is already active");
    active_=true; requests_.fill(MPI_REQUEST_NULL);
    for(int f=0;f<2*grid_.dimension;++f) if(neighbours_[f]!=MPI_PROC_NULL)
        mpi_check(MPI_Irecv(backend.receive_buffer(f),counts_[f],MPI_DOUBLE,neighbours_[f],200+(f^1),communicator_,&requests_[f]),"MPI_Irecv(boundary)");
    backend.pack_boundaries();
    backend.wait_outgoing(); // Packed device data / staged host data must be ready for MPI.
    for(int f=0;f<2*grid_.dimension;++f) if(neighbours_[f]!=MPI_PROC_NULL)
        mpi_check(MPI_Isend(backend.send_buffer(f),counts_[f],MPI_DOUBLE,neighbours_[f],200+f,communicator_,&requests_[6+f]),"MPI_Isend(boundary)");
}
void MpiRuntime::finish_exchange(Backend& backend) {
    if(!active_) throw std::logic_error("No boundary exchange is active");
    mpi_check(MPI_Waitall(static_cast<int>(requests_.size()),requests_.data(),MPI_STATUSES_IGNORE),"MPI_Waitall(boundaries)");
    backend.unpack_boundaries(); active_=false;
}
real MpiRuntime::maximum(real local) {
    real global=0; mpi_check(MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_MAX,communicator_),"MPI_Allreduce(max)"); return global;
}
real MpiRuntime::sum(real local) {
    real global=0; mpi_check(MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,communicator_),"MPI_Allreduce(sum)"); return global;
}
void MpiRuntime::barrier() { mpi_check(MPI_Barrier(communicator_),"MPI_Barrier(start)"); }
}
