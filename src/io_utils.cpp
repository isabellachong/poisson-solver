#include "io_utils.hpp"
#include "decomposition.hpp"
#include "problem.hpp"
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>
namespace poisson {
namespace {
std::string quoted(const std::string& text) {
    std::string result="\"";
    for(char ch:text) { if(ch=='"') result+='"'; result+=ch; }
    return result+'"';
}
}
void write_summary(std::ostream& out,const Config& c,const RunInfo& info,const Result& r) {
    const bool gpu=info.mode=="mpi-cuda";
    out << "mode,dimension,nx,ny,nz,unknowns,nonzeros,mpi_ranks,px,py,pz,gpus_used,max_ranks_per_gpu,"
        << "communication,overlap,block_size,problem,iterations,check_every,warmup,rtol,atol,converged,"
        << "residual_inf,rhs_inf,threshold,error_inf,error_rms,setup_seconds,solve_seconds,diagnostics_seconds,cell_updates_per_second,device_name\n";
    out << std::setprecision(17) << info.mode << ',' << c.dimension << ',' << c.points[0] << ',' << c.points[1] << ','
        << (c.dimension==3 ? c.points[2] : 1) << ',' << unknown_count(c) << ',' << nonzero_count(c) << ','
        << info.ranks << ',' << info.process_grid[0] << ',' << info.process_grid[1] << ',' << info.process_grid[2] << ','
        << info.gpus_used << ',' << info.max_ranks_per_gpu << ','
        << (gpu ? c.communication : (info.mode=="serial" ? "none" : "host")) << ','
        << (info.mode!="serial" && c.overlap) << ',' << (gpu ? c.block_size : 0) << ',' << c.problem << ',' << r.iterations << ',' << c.check_every << ','
        << c.warmup << ',' << c.rtol << ',' << c.atol << ',' << r.converged << ',' << r.residual_inf << ',' << r.rhs_inf << ',' << r.threshold << ','
        << r.error_inf << ',' << r.error_rms << ',' << r.setup_seconds << ',' << r.solve_seconds << ',' << r.diagnostics_seconds << ','
        << (r.solve_seconds>0 ? static_cast<double>(unknown_count(c))*r.iterations/r.solve_seconds : 0) << ',' << quoted(info.device_name) << '\n';
}
void write_solution(const Config& c,const RunInfo& info,const Grid& g,const std::vector<real>& values) {
    if(c.solution_prefix.empty()) return;
    std::ostringstream path; path << c.solution_prefix << ".rank" << std::setfill('0') << std::setw(4) << info.rank << ".csv";
    std::ofstream file(path.str());
    if(!file) throw std::runtime_error("Cannot create solution file: "+path.str());
    file << "global_i,global_j,global_k,x,y,z,u,exact\n" << std::setprecision(17);
    const auto r=owned_region(g);
    for(int k=r.begin[2];k<r.end[2];++k)
        for(int j=r.begin[1];j<r.end[1];++j)
            for(int i=r.begin[0];i<r.end[0];++i) {
                const real x=static_cast<real>(g.offset[0]+i)/(static_cast<real>(g.global[0])+1);
                const real y=static_cast<real>(g.offset[1]+j)/(static_cast<real>(g.global[1])+1);
                const real z=g.dimension==3 ? static_cast<real>(g.offset[2]+k)/(static_cast<real>(g.global[2])+1) : 0;
                file << g.offset[0]+i << ',' << g.offset[1]+j << ',' << (g.dimension==3 ? g.offset[2]+k : 0) << ','
                     << x << ',' << y << ',' << z << ',' << values[g.index(i,j,k)] << ','
                     << exact_solution(c.problem,g.dimension,x,y,z) << '\n';
            }
    file.close(); if(!file) throw std::runtime_error("Failed to write solution file: "+path.str());
}
}
