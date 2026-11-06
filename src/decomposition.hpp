#ifndef POISSON_DECOMPOSITION_HPP
#define POISSON_DECOMPOSITION_HPP
#include "types.hpp"
namespace poisson {
Partition partition_axis(int total, int parts, int coordinate);
Int3 choose_process_grid(const Config& config, int ranks);
Grid make_grid(const Config& config, Int3 dimensions, Int3 coordinates);
std::uint64_t unknown_count(const Config& config);
std::uint64_t nonzero_count(const Config& config);
}
#endif
