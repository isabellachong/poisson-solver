#ifndef POISSON_CUDA_UTILS_HPP
#define POISSON_CUDA_UTILS_HPP
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
namespace poisson {
inline void cuda_check(cudaError_t status,const char* operation) {
    if(status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
}
#define POISSON_CUDA(call) ::poisson::cuda_check((call),#call)
#endif
