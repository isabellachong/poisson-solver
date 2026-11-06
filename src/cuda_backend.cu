#include "backend.hpp"
#include "cuda_kernels.hpp"
#include "cuda_utils.hpp"
#include "problem.hpp"
#include <array>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>
namespace poisson {
namespace {
int launcher_local_rank(int visible) {
    if(visible==1) return 0;
    for(const char* name:{"POISSON_LOCAL_RANK","OMPI_COMM_WORLD_LOCAL_RANK","MPI_LOCALRANKID","MV2_COMM_WORLD_LOCAL_RANK","SLURM_LOCALID"}) {
        if(const char* text=std::getenv(name)) {
            std::size_t end=0; const int rank=std::stoi(text,&end);
            if(end!=std::string(text).size() || rank<0) throw std::runtime_error(std::string("Invalid node-local rank in ")+name);
            return rank;
        }
    }
    throw std::runtime_error("Cannot determine node-local rank before MPI_Init. Use a supported MPI launcher, set POISSON_LOCAL_RANK per process, or expose one allocated GPU per process.");
}
struct DeviceStorage {
    Grid grid;
    FaceMask active;
    Coefficients c;
    int block_size;
    bool staged;
    real *old_values=nullptr,*new_values=nullptr,*rhs=nullptr,*partial=nullptr,*scalar=nullptr,*host_scalar=nullptr;
    std::array<real*,6> send{},receive{},host_send{},host_receive{};
    cudaStream_t compute=nullptr,boundary_copy=nullptr;
    cudaEvent_t received_ready=nullptr;
    DeviceStorage(Grid g,FaceMask a,const Config& config):grid(g),active(a),c(coefficients(g)),
        block_size(config.block_size),staged(config.communication=="host") {}
    ~DeviceStorage() {
        if(compute) cudaStreamSynchronize(compute);
        if(boundary_copy) cudaStreamSynchronize(boundary_copy);
        if(old_values) cudaFree(old_values);
        if(new_values) cudaFree(new_values);
        if(rhs) cudaFree(rhs);
        if(partial) cudaFree(partial);
        if(scalar) cudaFree(scalar);
        if(host_scalar) cudaFreeHost(host_scalar);
        for(int f=0;f<6;++f) {
            if(send[f]) cudaFree(send[f]); if(receive[f]) cudaFree(receive[f]);
            if(host_send[f]) cudaFreeHost(host_send[f]); if(host_receive[f]) cudaFreeHost(host_receive[f]);
        }
        if(received_ready) cudaEventDestroy(received_ready);
        if(compute) cudaStreamDestroy(compute);
        if(boundary_copy) cudaStreamDestroy(boundary_copy);
    }
};
void allocate_device(real*& pointer,std::size_t count) { POISSON_CUDA(cudaMalloc(reinterpret_cast<void**>(&pointer),count*sizeof(real))); }
void allocate_host(real*& pointer,std::size_t count) { POISSON_CUDA(cudaMallocHost(reinterpret_cast<void**>(&pointer),count*sizeof(real))); }

class CudaBackend final : public Backend {
    // Owning storage also cleans up allocations if construction fails partway.
    std::unique_ptr<DeviceStorage> p_;
public:
    CudaBackend(Grid g,FaceMask active,const Fields& initial,const Config& config)
        :p_(std::make_unique<DeviceStorage>(g,active,config)) {
        int device=0; cudaDeviceProp properties{};
        POISSON_CUDA(cudaGetDevice(&device)); POISSON_CUDA(cudaGetDeviceProperties(&properties,device));
        if(config.block_size>properties.maxThreadsPerBlock) throw std::invalid_argument("block-size exceeds this GPU's limit");
        auto& p=*p_;
        POISSON_CUDA(cudaStreamCreateWithFlags(&p.compute,cudaStreamNonBlocking));
        POISSON_CUDA(cudaStreamCreateWithFlags(&p.boundary_copy,cudaStreamNonBlocking));
        POISSON_CUDA(cudaEventCreateWithFlags(&p.received_ready,cudaEventDisableTiming));
        allocate_device(p.old_values,g.size()); allocate_device(p.new_values,g.size()); allocate_device(p.rhs,g.size());
        allocate_device(p.partial,reduction_blocks(g)); allocate_device(p.scalar,1); allocate_host(p.host_scalar,1);
        for(int f=0;f<2*g.dimension;++f) if(active[f]) {
            const auto count=face_size(g,f);
            allocate_device(p.send[f],count); allocate_device(p.receive[f],count);
            if(p.staged) { allocate_host(p.host_send[f],count); allocate_host(p.host_receive[f],count); }
        }
        reset(initial);
    }
    const Grid& grid() const override { return p_->grid; }
    bool uses_gpu() const override { return true; }
    void reset(const Fields& initial) override {
        auto& p=*p_;
        if(initial.solution.size()!=p.grid.size() || initial.rhs.size()!=p.grid.size()) throw std::invalid_argument("Initial arrays do not match local grid");
        POISSON_CUDA(cudaStreamSynchronize(p.compute)); POISSON_CUDA(cudaStreamSynchronize(p.boundary_copy));
        const auto bytes=p.grid.size()*sizeof(real);
        POISSON_CUDA(cudaMemcpy(p.old_values,initial.solution.data(),bytes,cudaMemcpyHostToDevice));
        POISSON_CUDA(cudaMemcpy(p.new_values,initial.solution.data(),bytes,cudaMemcpyHostToDevice));
        POISSON_CUDA(cudaMemcpy(p.rhs,initial.rhs.data(),bytes,cudaMemcpyHostToDevice));
    }
    void update(Region region) override {
        auto& p=*p_;
        launch_update(p.old_values,p.new_values,p.rhs,p.grid,p.c,region,p.block_size,p.compute);
    }
    void pack_boundaries() override {
        auto& p=*p_;
        for(int f=0;f<2*p.grid.dimension;++f) if(p.active[f]) {
            launch_pack(p.old_values,p.send[f],p.grid,f,p.block_size,p.boundary_copy);
            if(p.staged) POISSON_CUDA(cudaMemcpyAsync(p.host_send[f],p.send[f],face_size(p.grid,f)*sizeof(real),cudaMemcpyDeviceToHost,p.boundary_copy));
        }
    }
    void wait_outgoing() override { POISSON_CUDA(cudaStreamSynchronize(p_->boundary_copy)); }
    real* send_buffer(int face) override { return p_->staged ? p_->host_send[face] : p_->send[face]; }
    real* receive_buffer(int face) override { return p_->staged ? p_->host_receive[face] : p_->receive[face]; }
    void unpack_boundaries() override {
        auto& p=*p_;
        for(int f=0;f<2*p.grid.dimension;++f) if(p.active[f]) {
            if(p.staged) POISSON_CUDA(cudaMemcpyAsync(p.receive[f],p.host_receive[f],face_size(p.grid,f)*sizeof(real),cudaMemcpyHostToDevice,p.boundary_copy));
            launch_unpack(p.old_values,p.receive[f],p.grid,f,p.block_size,p.boundary_copy);
        }
        // Interior reads only owned data; boundary updates wait for received values.
        POISSON_CUDA(cudaEventRecord(p.received_ready,p.boundary_copy));
        POISSON_CUDA(cudaStreamWaitEvent(p.compute,p.received_ready,0));
    }
    void finish_iteration() override {
        POISSON_CUDA(cudaStreamSynchronize(p_->compute));
        std::swap(p_->old_values,p_->new_values);
    }
    real local_residual() override {
        auto& p=*p_;
        launch_residual(p.old_values,p.rhs,p.grid,p.c,p.partial,p.scalar,p.compute);
        POISSON_CUDA(cudaMemcpyAsync(p.host_scalar,p.scalar,sizeof(real),cudaMemcpyDeviceToHost,p.compute));
        POISSON_CUDA(cudaStreamSynchronize(p.compute));
        return *p.host_scalar;
    }
    std::vector<real> download() override {
        auto& p=*p_;
        POISSON_CUDA(cudaStreamSynchronize(p.compute)); POISSON_CUDA(cudaStreamSynchronize(p.boundary_copy));
        std::vector<real> values(p.grid.size());
        POISSON_CUDA(cudaMemcpy(values.data(),p.old_values,values.size()*sizeof(real),cudaMemcpyDeviceToHost));
        return values;
    }
};
}
Device prepare_cuda_device() {
    int count=0; POISSON_CUDA(cudaGetDeviceCount(&count));
    if(count<1) throw std::runtime_error("No allocated CUDA GPU is visible");
    const int local_rank=launcher_local_rank(count);
    const int selected=local_rank%count;
    POISSON_CUDA(cudaSetDevice(selected));
    cudaDeviceProp properties{}; POISSON_CUDA(cudaGetDeviceProperties(&properties,selected));
    std::ostringstream uuid; uuid << std::hex << std::setfill('0');
    for(unsigned char byte:properties.uuid.bytes) uuid << std::setw(2) << static_cast<unsigned>(byte);
    return {selected,properties.name,uuid.str()};
}
std::unique_ptr<Backend> make_cuda_backend(Grid grid,FaceMask active,const Fields& initial,const Config& config) {
    return std::make_unique<CudaBackend>(grid,active,initial,config);
}
}
