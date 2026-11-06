#ifndef POISSON_SOLVER_HPP
#define POISSON_SOLVER_HPP
#include "runtime.hpp"
namespace poisson {
Result solve(const Config& config, Fields initial, Backend& backend, Runtime& runtime,
             const RunInfo& info, double setup_start);
}
#endif
