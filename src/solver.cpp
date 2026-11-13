#include "solver.hpp"
#include "problem.hpp"
#include "decomposition.hpp"
#include "io_utils.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace poisson {
Result solve(const Config& c, Fields initial, Backend& backend, Runtime& runtime,
             const RunInfo& info, double setup_start) {
    const auto& g=backend.grid();
    const auto interior=interior_region(g);
    const auto boundaries=boundary_regions(g);
    // Jacobi update; interior points are updated first, then boundaries. 
    auto iteration=[&] {
        if(!runtime.distributed_execution()) backend.update(owned_region(g));
        else {
            runtime.begin_exchange(backend);
            if(c.overlap) backend.update(interior);
            runtime.finish_exchange(backend);
            if(!c.overlap) backend.update(interior);
            for(const auto& r:boundaries) backend.update(r);
        }
        backend.finish_iteration(); // Completes sweep and swaps old/new solution arrays. 
    };
    auto residual=[&] {
        // The owned solution has advanced after a swap; refresh neighbouring values.
        runtime.begin_exchange(backend); runtime.finish_exchange(backend);
        const real value=runtime.maximum(backend.local_residual());
        if(!std::isfinite(value)) throw std::runtime_error("Nonfinite global residual");
        return value;
    };
    Result result;
    result.rhs_inf=runtime.maximum(initial.rhs_inf);
    result.threshold=c.atol+c.rtol*result.rhs_inf;
    if(!std::isfinite(result.threshold)) throw std::invalid_argument("Residual tolerance overflows");
    // Run warmpup iterations to avoid measuring initialization overhead.
    for(int k=0;k<c.warmup;++k) iteration();
    if(c.warmup) backend.reset(initial);
    initial=Fields{}; // Release initialization arrays before the measured solve.
    bool current_residual=false;
    if(c.check_every) { result.residual_inf=residual(); result.converged=result.residual_inf<=result.threshold; current_residual=true; }
    const double setup_elapsed=runtime.now()-setup_start;
    runtime.barrier();
    // Solve: iterate until convergence or maximum iterations.
    const double start=runtime.now();
    while(result.iterations<c.iterations && !(c.check_every && result.converged)) {
        iteration(); ++result.iterations; current_residual=false;
        if(c.check_every && (result.iterations%c.check_every==0 || result.iterations==c.iterations)) {
            result.residual_inf=residual(); current_residual=true;
            result.converged=result.residual_inf<=result.threshold;
        }
    }
    const double solve_elapsed=runtime.now()-start;
    // Compute error against exact solution and write output.
    const double diagnostics_start=runtime.now();
    if(!current_residual) result.residual_inf=residual();
    result.converged=result.residual_inf<=result.threshold;
    const auto values=backend.download();
    const auto r=owned_region(g);
    real local_max=0, local_squares=0;
    for(int k=r.begin[2];k<r.end[2];++k)
        for(int j=r.begin[1];j<r.end[1];++j)
            for(int i=r.begin[0];i<r.end[0];++i) {
                const real x=static_cast<real>(g.offset[0]+i)/(static_cast<real>(g.global[0])+1);
                const real y=static_cast<real>(g.offset[1]+j)/(static_cast<real>(g.global[1])+1);
                const real z=g.dimension==3 ? static_cast<real>(g.offset[2]+k)/(static_cast<real>(g.global[2])+1) : 0;
                const real error=values[g.index(i,j,k)]-exact_solution(c.problem,g.dimension,x,y,z);
                if(!std::isfinite(error)) throw std::runtime_error("Nonfinite solution error");
                local_max=std::max(local_max,std::abs(error)); local_squares+=error*error;
            }
    result.error_inf=runtime.maximum(local_max);
    result.error_rms=std::sqrt(runtime.sum(local_squares)/static_cast<real>(unknown_count(c)));
    write_solution(c,info,g,values);
    const double diagnostics_elapsed=runtime.now()-diagnostics_start;
    result.setup_seconds=runtime.maximum(setup_elapsed);
    result.solve_seconds=runtime.maximum(solve_elapsed);
    result.diagnostics_seconds=runtime.maximum(diagnostics_elapsed);
    return result;
}
}
