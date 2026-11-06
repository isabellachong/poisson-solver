#include "backend.hpp"
#include "config.hpp"
#include "decomposition.hpp"
#include "io_utils.hpp"
#include "problem.hpp"
#include "solver.hpp"
#include <iostream>
#include <stdexcept>
#include <utility>
int main(int argc,char* argv[]) {
    try {
        if(poisson::help_requested(argc,argv)) { poisson::print_usage(std::cout); return 0; }
        const auto c=poisson::parse_arguments(argc,argv);
        // Reject explicit distributed settings for the serial executable.
        poisson::choose_process_grid(c,1);
        poisson::SerialRuntime runtime;
        const double start=runtime.now();
        const auto grid=poisson::make_grid(c,{1,1,1},{0,0,0});
        auto fields=poisson::initialize_fields(grid,c.problem);
        auto backend=poisson::make_cpu_backend(grid,{},fields);
        poisson::RunInfo info;
        const auto result=poisson::solve(c,std::move(fields),*backend,runtime,info,start);
        poisson::write_summary(std::cout,c,info,result);
        return c.check_every && !result.converged ? 2 : 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
