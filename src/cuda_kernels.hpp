#ifndef POISSON_CUDA_KERNELS_HPP
#define POISSON_CUDA_KERNELS_HPP
#include "types.hpp"
#include <cuda_runtime_api.h>
namespace poisson {
void launch_update(const real* old_values,real* new_values,const real* rhs,Grid grid,Coefficients coefficients,
                   Region region,int block_size,cudaStream_t stream);
void launch_pack(const real* values,real* buffer,Grid grid,int face,int block_size,cudaStream_t stream);
void launch_unpack(real* values,const real* buffer,Grid grid,int face,int block_size,cudaStream_t stream);
std::size_t reduction_blocks(const Grid& grid);
void launch_residual(const real* values,const real* rhs,Grid grid,Coefficients coefficients,
                     real* partial,real* result,cudaStream_t stream);
}
#endif
