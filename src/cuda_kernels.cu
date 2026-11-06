#include "cuda_kernels.hpp"
#include "cuda_utils.hpp"
#include "problem.hpp"
#include <math_constants.h>
#include <algorithm>
namespace poisson {
namespace {
constexpr int reduce_threads=256;
unsigned int blocks_for(std::size_t count,int block_size) {
    return static_cast<unsigned int>(std::min<std::size_t>(65535,(count+block_size-1)/block_size));
}
__global__ void update_kernel(const real* __restrict__ old_values,real* __restrict__ next,
                              const real* __restrict__ rhs,Grid g,Coefficients c,Region r,std::size_t count) {
    const auto width=static_cast<std::size_t>(r.end[0]-r.begin[0]);
    const auto height=static_cast<std::size_t>(r.end[1]-r.begin[1]);
    for(std::size_t q=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
        q<count;q+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
        const int i=r.begin[0]+static_cast<int>(q%width);
        const int j=r.begin[1]+static_cast<int>((q/width)%height);
        const int k=r.begin[2]+static_cast<int>(q/(width*height));
        const auto index=g.index(i,j,k);
        next[index]=stencil_value(old_values,rhs,index,g,c);
    }
}
__global__ void pack_kernel(const real* values,real* buffer,Grid g,int face,std::size_t count) {
    for(std::size_t q=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
        q<count;q+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
        int i,j,k; face_point(g,face,q,false,i,j,k); buffer[q]=values[g.index(i,j,k)];
    }
}
__global__ void unpack_kernel(real* values,const real* buffer,Grid g,int face,std::size_t count) {
    for(std::size_t q=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
        q<count;q+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
        int i,j,k; face_point(g,face,q,true,i,j,k); values[g.index(i,j,k)]=buffer[q];
    }
}
__global__ void residual_kernel(const real* values,const real* rhs,Grid g,Coefficients c,
                                real* partial,std::size_t count) {
    __shared__ real maxima[reduce_threads];
    real value=0;
    for(std::size_t q=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
        q<count;q+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
        const int i=static_cast<int>(q%g.n[0])+1;
        const int j=static_cast<int>((q/g.n[0])%g.n[1])+1;
        const int k=g.dimension==3 ? static_cast<int>(q/(static_cast<std::size_t>(g.n[0])*g.n[1]))+1 : 0;
        const real residual=stencil_residual(values,rhs,g.index(i,j,k),g,c);
        value=fmax(value,isfinite(residual) ? fabs(residual) : CUDART_INF);
    }
    maxima[threadIdx.x]=value;
    __syncthreads();
    for(int stride=reduce_threads/2;stride>0;stride/=2) {
        if(threadIdx.x<stride) maxima[threadIdx.x]=fmax(maxima[threadIdx.x],maxima[threadIdx.x+stride]);
        __syncthreads();
    }
    if(threadIdx.x==0) partial[blockIdx.x]=maxima[0];
}
__global__ void finish_reduction(const real* partial,real* result,std::size_t count) {
    __shared__ real maxima[reduce_threads];
    real value=0;
    for(std::size_t q=threadIdx.x;q<count;q+=blockDim.x) value=fmax(value,partial[q]);
    maxima[threadIdx.x]=value;
    __syncthreads();
    for(int stride=reduce_threads/2;stride>0;stride/=2) {
        if(threadIdx.x<stride) maxima[threadIdx.x]=fmax(maxima[threadIdx.x],maxima[threadIdx.x+stride]);
        __syncthreads();
    }
    if(threadIdx.x==0) *result=maxima[0];
}
}
void launch_update(const real* old_values,real* next,const real* rhs,Grid g,Coefficients c,
                   Region r,int block_size,cudaStream_t stream) {
    const auto count=r.size(); if(!count) return;
    update_kernel<<<blocks_for(count,block_size),block_size,0,stream>>>(old_values,next,rhs,g,c,r,count);
    POISSON_CUDA(cudaGetLastError());
}
void launch_pack(const real* values,real* buffer,Grid g,int face,int block_size,cudaStream_t stream) {
    const auto count=face_size(g,face);
    pack_kernel<<<blocks_for(count,block_size),block_size,0,stream>>>(values,buffer,g,face,count);
    POISSON_CUDA(cudaGetLastError());
}
void launch_unpack(real* values,const real* buffer,Grid g,int face,int block_size,cudaStream_t stream) {
    const auto count=face_size(g,face);
    unpack_kernel<<<blocks_for(count,block_size),block_size,0,stream>>>(values,buffer,g,face,count);
    POISSON_CUDA(cudaGetLastError());
}
std::size_t reduction_blocks(const Grid& g) { return std::min<std::size_t>(4096,(g.owned()+reduce_threads-1)/reduce_threads); }
void launch_residual(const real* values,const real* rhs,Grid g,Coefficients c,
                     real* partial,real* result,cudaStream_t stream) {
    const auto blocks=reduction_blocks(g);
    residual_kernel<<<static_cast<unsigned>(blocks),reduce_threads,0,stream>>>(values,rhs,g,c,partial,g.owned());
    POISSON_CUDA(cudaGetLastError());
    finish_reduction<<<1,reduce_threads,0,stream>>>(partial,result,blocks);
    POISSON_CUDA(cudaGetLastError());
}
}
