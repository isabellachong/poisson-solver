#ifndef POISSON_IO_UTILS_HPP
#define POISSON_IO_UTILS_HPP
#include "types.hpp"
#include <iosfwd>
namespace poisson {
void write_summary(std::ostream& out,const Config& config,const RunInfo& info,const Result& result);
void write_solution(const Config& config,const RunInfo& info,const Grid& grid,const std::vector<real>& values);
}
#endif
