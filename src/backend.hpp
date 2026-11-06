#ifndef POISSON_BACKEND_HPP
#define POISSON_BACKEND_HPP
#include "types.hpp"
#include <memory>
namespace poisson {
class Backend {
public:
    virtual ~Backend()=default;
    virtual const Grid& grid() const=0;
    virtual void reset(const Fields& fields)=0;
    virtual void update(Region region)=0;
    virtual void pack_boundaries()=0;
    virtual void wait_outgoing()=0;
    virtual real* send_buffer(int face)=0;
    virtual real* receive_buffer(int face)=0;
    virtual void unpack_boundaries()=0;
    virtual void finish_iteration()=0;
    virtual real local_residual()=0;
    virtual std::vector<real> download()=0;
    virtual bool uses_gpu() const { return false; }
};
std::unique_ptr<Backend> make_cpu_backend(Grid grid, FaceMask active, const Fields& initial);
// CUDA entry points are linked into poisson_mpi in a CUDA-enabled build.
Device prepare_cuda_device();
std::unique_ptr<Backend> make_cuda_backend(Grid grid, FaceMask active, const Fields& initial, const Config& config);
}
#endif
