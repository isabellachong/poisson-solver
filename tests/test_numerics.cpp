#include "backend.hpp"
#include "config.hpp"
#include "decomposition.hpp"
#include "problem.hpp"
#include "solver.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
template<class F> void rejects(F function) {
    bool rejected=false; try { function(); } catch(const std::exception&) { rejected=true; }
    require(rejected,"Invalid input was accepted");
}
poisson::Config config(int dimension,int nx,int ny,int nz=1) {
    poisson::Config c; c.dimension=dimension; c.points={nx,ny,nz}; c.warmup=0; return c;
}
void decomposition() {
    auto c=config(2,64,64);
    require(poisson::choose_process_grid(c,4)==poisson::Int3{2,2,1},"Expected 2x2 decomposition");
    require(poisson::choose_process_grid(c,256)==poisson::Int3{16,16,1},"Expected 16x16 decomposition");
    rejects([&] { poisson::choose_process_grid(c,2); });
    c.process_grid={4,1,1}; rejects([&] { poisson::choose_process_grid(c,4); });
    c=config(3,16,16,16);
    require(poisson::choose_process_grid(c,8)==poisson::Int3{2,2,2},"Expected 2x2x2 decomposition");
    require(poisson::unknown_count(c)==4096 && poisson::nonzero_count(c)==27136,"3D sparse system count is wrong");
    c=config(2,64,64);
    require(poisson::nonzero_count(c)==20224,"2D sparse system count is wrong");
    const auto first=poisson::partition_axis(7,2,0),last=poisson::partition_axis(7,2,1);
    require(first.count==4 && first.offset==0 && last.count==3 && last.offset==4,"Uneven partition is wrong");
}
void stencil() {
    for(int dimension:{2,3}) {
        const auto c=config(dimension,7,5,4);
        const auto g=poisson::make_grid(c,{1,1,1},{0,0,0});
        auto initial=poisson::initialize_fields(g,"polynomial");
        require(initial.solution[g.index(0,3,dimension==3 ? 2 : 0)]>0,"Nonzero physical boundary missing");
        const auto region=poisson::owned_region(g);
        for(int k=region.begin[2];k<region.end[2];++k)
            for(int j=1;j<=g.n[1];++j)
                for(int i=1;i<=g.n[0];++i) {
                    const double x=i/8.0,y=j/6.0,z=dimension==3 ? k/5.0 : 0;
                    initial.solution[g.index(i,j,k)]=poisson::exact_solution("polynomial",dimension,x,y,z);
                }
        auto backend=poisson::make_cpu_backend(g,{},initial);
        require(backend->local_residual()<2e-12,"Discrete polynomial residual is wrong (spacing, sign, or boundary)");
        const auto subgrid=poisson::make_grid(c,{2,2,dimension==3 ? 2 : 1},{1,1,dimension==3 ? 1 : 0});
        const auto sub=poisson::initialize_fields(subgrid,"polynomial");
        require(sub.solution[subgrid.index(0,0,dimension==3 ? 0 : 0)]==0,"Internal rank boundary treated as prescribed physical boundary");
    }
}
void regions() {
    for(const auto& c:{config(2,5,4),config(2,1,2),config(3,4,3,5),config(3,2,1,2)}) {
        const auto g=poisson::make_grid(c,{1,1,1},{0,0,0});
        std::vector<int> coverage(g.size(),0);
        auto mark=[&](poisson::Region r) {
            if(!r.size()) return;
            for(int k=r.begin[2];k<r.end[2];++k) for(int j=r.begin[1];j<r.end[1];++j) for(int i=r.begin[0];i<r.end[0];++i)
                ++coverage[g.index(i,j,k)];
        };
        mark(poisson::interior_region(g));
        for(const auto& r:poisson::boundary_regions(g)) mark(r);
        const auto owned=poisson::owned_region(g);
        std::size_t count=0;
        for(int k=owned.begin[2];k<owned.end[2];++k) for(int j=1;j<=g.n[1];++j) for(int i=1;i<=g.n[0];++i) {
            require(coverage[g.index(i,j,k)]==1,"A point was skipped or updated twice"); ++count;
        }
        require(count==g.owned(),"Owned point count is wrong");
    }
}
poisson::Int3 coordinates(int rank,poisson::Int3 p) {
    return {rank/(p[1]*p[2]),(rank/p[2])%p[1],rank%p[2]};
}
int rank_of(poisson::Int3 xyz,poisson::Int3 p) { return (xyz[0]*p[1]+xyz[1])*p[2]+xyz[2]; }
void exchange(int dimension,bool thin=false) {
    const auto c=thin ? config(dimension,3,3,3) : config(dimension,9,7,5);
    const poisson::Int3 shape{2,2,dimension==3 ? 2 : 1};
    const int ranks=shape[0]*shape[1]*shape[2];
    const auto whole=poisson::make_grid(c,{1,1,1},{0,0,0});
    const auto full_initial=poisson::initialize_fields(whole,"polynomial");
    for(bool overlap:{false,true}) {
        auto reference=poisson::make_cpu_backend(whole,{},full_initial);
        std::vector<std::unique_ptr<poisson::Backend>> local;
        std::vector<std::array<int,6>> neighbours(ranks);
        for(int rank=0;rank<ranks;++rank) {
            const auto xyz=coordinates(rank,shape);
            poisson::FaceMask active{}; neighbours[rank].fill(-1);
            for(int f=0;f<2*dimension;++f) {
                auto other=xyz; other[f/2]+=f%2 ? 1 : -1;
                active[f]=other[f/2]>=0 && other[f/2]<shape[f/2];
                if(active[f]) neighbours[rank][f]=rank_of(other,shape);
            }
            const auto g=poisson::make_grid(c,shape,xyz);
            local.push_back(poisson::make_cpu_backend(g,active,poisson::initialize_fields(g,"polynomial")));
        }
        auto transfer=[&] {
            for(auto& backend:local) backend->pack_boundaries();
            for(int rank=0;rank<ranks;++rank) for(int f=0;f<2*dimension;++f) if(neighbours[rank][f]>=0) {
                const int source=neighbours[rank][f];
                std::copy_n(local[source]->send_buffer(f^1),poisson::face_size(local[rank]->grid(),f),local[rank]->receive_buffer(f));
            }
        };
        for(int iteration=0;iteration<9;++iteration) {
            transfer();
            if(overlap) for(auto& backend:local) backend->update(poisson::interior_region(backend->grid()));
            for(auto& backend:local) {
                backend->unpack_boundaries();
                if(!overlap) backend->update(poisson::interior_region(backend->grid()));
                for(const auto& r:poisson::boundary_regions(backend->grid())) backend->update(r);
                backend->finish_iteration();
            }
            reference->update(poisson::owned_region(whole)); reference->finish_iteration();
        }
        const auto expected=reference->download();
        transfer();
        double maximum=0;
        for(auto& backend:local) {
            backend->unpack_boundaries(); maximum=std::max(maximum,backend->local_residual());
            const auto& g=backend->grid(); const auto values=backend->download(); const auto owned=poisson::owned_region(g);
            for(int k=owned.begin[2];k<owned.end[2];++k) for(int j=1;j<=g.n[1];++j) for(int i=1;i<=g.n[0];++i)
                require(std::abs(values[g.index(i,j,k)]-expected[whole.index(g.offset[0]+i,g.offset[1]+j,g.offset[2]+k)])<2e-13,
                        "Multidimensional boundary exchange disagrees with full-grid solve");
        }
        require(std::abs(maximum-reference->local_residual())<1e-10,"Residual uses incorrect neighbouring values");
    }
}
void convergence() {
    for(int dimension:{2,3}) {
        double errors[2]{};
        for(int level=0;level<2;++level) {
            const int n=level==0 ? 7 : 15;
            auto c=config(dimension,n,n,n); c.problem="sine"; c.iterations=10000; c.check_every=10; c.rtol=1e-10;
            poisson::SerialRuntime runtime; const auto start=runtime.now();
            const auto g=poisson::make_grid(c,{1,1,1},{0,0,0});
            auto fields=poisson::initialize_fields(g,c.problem); auto backend=poisson::make_cpu_backend(g,{},fields);
            const auto result=poisson::solve(c,std::move(fields),*backend,runtime,{},start);
            require(result.converged && result.residual_inf<=result.threshold,"Serial solver did not converge");
            errors[level]=result.error_inf;
        }
        require(errors[0]/errors[1]>3.9 && errors[0]/errors[1]<4.2,"Expected second-order spatial accuracy");
    }
}
void configuration() {
    auto c=config(3,8,7,6); poisson::validate_config(c);
    c.communication="host"; poisson::validate_config(c);
    c.points[0]=0; rejects([&]{poisson::validate_config(c);});
    c=config(2,8,7); c.rtol=std::numeric_limits<double>::quiet_NaN(); rejects([&]{poisson::validate_config(c);});
    c=config(2,8,7); c.process_grid[2]=2; rejects([&]{poisson::validate_config(c);});
    c=config(2,8,7); c.block_size=33; rejects([&]{poisson::validate_config(c);});
    c=config(2,8,7); c.points[2]=0; poisson::validate_config(c); // Unused nz is ignored in 2D.
}
}
int main(int argc,char* argv[]) {
    try {
        if(argc!=2) throw std::invalid_argument("Expected one named check");
        const std::string check=argv[1];
        if(check=="decomposition") decomposition(); else if(check=="stencil") stencil();
        else if(check=="regions") regions(); else if(check=="exchange2d") { exchange(2); exchange(2,true); }
        else if(check=="exchange3d") { exchange(3); exchange(3,true); } else if(check=="convergence") convergence();
        else if(check=="configuration") configuration(); else throw std::invalid_argument("Unknown check");
        std::cout << "Passed: " << check << '\n'; return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
