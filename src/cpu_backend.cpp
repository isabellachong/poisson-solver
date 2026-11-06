#include "backend.hpp"
#include "problem.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
namespace poisson {
namespace {
class CpuBackend final : public Backend {
    Grid grid_;
    FaceMask active_;
    Coefficients c_;
    std::vector<real> old_, next_, rhs_;
    std::array<std::vector<real>,6> sends_,receives_;
public:
    CpuBackend(Grid g, FaceMask active, const Fields& initial):grid_(g),active_(active),c_(coefficients(g)) {
        for(int f=0;f<2*g.dimension;++f) if(active[f]) { sends_[f].resize(face_size(g,f)); receives_[f].resize(face_size(g,f)); }
        reset(initial);
    }
    const Grid& grid() const override { return grid_; }
    void reset(const Fields& fields) override {
        if(fields.solution.size()!=grid_.size() || fields.rhs.size()!=grid_.size()) throw std::invalid_argument("Initial arrays do not match local grid");
        old_=fields.solution; next_=fields.solution; rhs_=fields.rhs;
    }
    void update(Region r) override {
        if(!r.size()) return;
        for(int k=r.begin[2];k<r.end[2];++k)
            for(int j=r.begin[1];j<r.end[1];++j)
                for(int i=r.begin[0];i<r.end[0];++i) {
                    const auto q=grid_.index(i,j,k);
                    next_[q]=stencil_value(old_.data(),rhs_.data(),q,grid_,c_);
                }
    }
    void pack_boundaries() override {
        for(int f=0;f<2*grid_.dimension;++f) if(active_[f])
            for(std::size_t q=0;q<sends_[f].size();++q) {
                int i,j,k; face_point(grid_,f,q,false,i,j,k);
                sends_[f][q]=old_[grid_.index(i,j,k)];
            }
    }
    void wait_outgoing() override {}
    real* send_buffer(int f) override { return sends_[f].data(); }
    real* receive_buffer(int f) override { return receives_[f].data(); }
    void unpack_boundaries() override {
        for(int f=0;f<2*grid_.dimension;++f) if(active_[f])
            for(std::size_t q=0;q<receives_[f].size();++q) {
                int i,j,k; face_point(grid_,f,q,true,i,j,k);
                old_[grid_.index(i,j,k)]=receives_[f][q];
            }
    }
    void finish_iteration() override { std::swap(old_,next_); }
    real local_residual() override {
        real maximum=0;
        const auto r=owned_region(grid_);
        for(int k=r.begin[2];k<r.end[2];++k)
            for(int j=r.begin[1];j<r.end[1];++j)
                for(int i=r.begin[0];i<r.end[0];++i) {
                    const real residual=stencil_residual(old_.data(),rhs_.data(),grid_.index(i,j,k),grid_,c_);
                    if(!std::isfinite(residual)) return std::numeric_limits<real>::infinity();
                    maximum=std::max(maximum,std::abs(residual));
                }
        return maximum;
    }
    std::vector<real> download() override { return old_; }
};
}
std::unique_ptr<Backend> make_cpu_backend(Grid g, FaceMask active, const Fields& initial) {
    return std::make_unique<CpuBackend>(g,active,initial);
}
}
